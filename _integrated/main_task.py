"""
main_task.py — 통합 시나리오
================================================================
자연어 프롬프트(예: "사과 잡아와") → 물건 좌표 → 출발 →
A* 내비 → (도로) 비전으로 신호등 초록 인지 → 횡단보도 통과(초록 연장) →
물건 위치 도착 → top-down IK 파지(닫힘 시 weld 부착) →
최종 위치(0,-19) 복귀 운반 → 내려놓기.

- 보행 : jaemin holonomic foot-placement trot
- 파지 : 6-DoF(위치+접근축) damped-LS IK + weld 부착
- 비전 : 로봇 시점 카메라 렌더에서 초록 픽셀 검출로 신호등 인지
- 맵/교통 : 친구(taekyeong) pathfinding / traffic / cars 재사용
================================================================
"""
import os
import sys
import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import mujoco
import mujoco.viewer

# ── 친구 맵 모듈 경로 추가 ────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_TK = os.path.join(_HERE, "..", "_taekyeong")
if _TK not in sys.path:
    sys.path.insert(0, _TK)
import pathfinding          # noqa: E402
import traffic              # noqa: E402
import cars                 # noqa: E402

# 신호등 초록 시간 연장 (요청사항) — 로봇이 도로를 완전히 건널 시간 확보
traffic.GREEN_DURATION = 16.0
traffic.TRAFFIC_CYCLE  = 24.0

# 초록 플랫폼(0.2m 턱)을 A* 장애물에 추가 → 경로가 플랫폼을 피해 남쪽에서 접근
pathfinding._OBSTACLES += [
    (-22.0, 23.0, 2.0, 1.0),   # green_01
    ( 14.0, 13.0, 1.0, 2.0),   # green_02
    (-11.0, -2.0, 1.0, 2.0),   # green_03
]
pathfinding._GRID_BASE = pathfinding._build_grid()


# ============================================================
# 0. 모델 로드
# ============================================================
SCENE_XML = os.path.join(_HERE, "scene_task.xml")
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data  = mujoco.MjData(model)
ik_model = mujoco.MjModel.from_xml_path(SCENE_XML)
ik_data  = mujoco.MjData(ik_model)


# ============================================================
# 1. 유틸
# ============================================================
def get_act(name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid == -1:
        raise ValueError(f"Actuator not found: {name}")
    return aid

def jnt_qadr(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_qposadr[jid]

def jnt_dadr(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_dofadr[jid]

def jnt_range(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = model.jnt_range[jid]
    return float(lo), float(hi)

def body_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

def eq_id(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

def find_free_qpos():
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[j]
    raise RuntimeError("free joint not found")

def quat_to_rpy(q):
    w, x, y, z = q
    roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(float(np.clip(2*(w*y-z*x), -1, 1)))
    yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return roll, pitch, yaw

def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))

def wrap_angle(a):
    return (a + math.pi) % (2*math.pi) - math.pi


# ============================================================
# 2. 액추에이터 / 인덱스
# ============================================================
fl_hx, fl_hy, fl_kn = get_act("fl_hx"), get_act("fl_hy"), get_act("fl_kn")
fr_hx, fr_hy, fr_kn = get_act("fr_hx"), get_act("fr_hy"), get_act("fr_kn")
hl_hx, hl_hy, hl_kn = get_act("hl_hx"), get_act("hl_hy"), get_act("hl_kn")
hr_hx, hr_hy, hr_kn = get_act("hr_hx"), get_act("hr_hy"), get_act("hr_kn")

arm_sh0 = get_act("arm_sh0"); arm_sh1 = get_act("arm_sh1")
arm_el0 = get_act("arm_el0"); arm_el1 = get_act("arm_el1")
arm_wr0 = get_act("arm_wr0"); arm_wr1 = get_act("arm_wr1")
arm_f1x = get_act("arm_f1x")

BASE    = find_free_qpos()
BODY_ID = body_id("body")
EE_BODY_ID = body_id("arm_link_fngr")
EE_LOCAL_OFFSET = np.array([0.10, 0.0, 0.0])

ARM_JNT = ["arm_sh0","arm_sh1","arm_el0","arm_el1","arm_wr0","arm_wr1"]
ARM_ACT_IDS  = [arm_sh0, arm_sh1, arm_el0, arm_el1, arm_wr0, arm_wr1]
ARM_QPOS_ADR = [jnt_qadr(n) for n in ARM_JNT]
ARM_DOF_ADR  = [jnt_dadr(n) for n in ARM_JNT]
ARM_RANGE    = [jnt_range(n) for n in ARM_JNT]

GRASP_AXIS = np.array([0.0, 0.0, -1.0])


# ============================================================
# 3. 물건 레지스트리 (이름 → 정보)
# ============================================================
@dataclass
class Obj:
    key: str
    body: str
    free: str
    weld: str
    start: np.ndarray          # 시작 위치 (받침 위)
    stop:  Tuple[float, float] # 파지 접근 정지점 (바닥)
    body_id: int = 0
    qadr: int = 0
    dadr: int = 0
    eqid: int = 0

# 한국어/영어 이름 → key  (실물 스캔 사물)
NAME_ALIASES = {
    "mario": "mario", "마리오": "mario",
    "yoshi": "yoshi", "요시": "yoshi",
    "android": "android", "안드로이드": "android", "로봇": "android",
    "dino": "dino", "공룡": "dino", "다이노": "dino", "디노": "dino",
    "elephant": "elephant", "코끼리": "elephant",
    "dog": "dog", "강아지": "dog", "개": "dog", "puppy": "dog",
}

# 실물 사물 base 가 z=0.20(플랫폼 윗면)에 놓임.
# stop = staging point (사물 남쪽/서쪽 ~1.6m, 플랫폼 인플레이션 밖). 최종 0.62m 접근은 APPROACH.
OBJECTS = {
    # green_01 (북서, 횡단)
    "mario":    Obj("mario",    "obj_mario",    "mario_free",    "grab_mario",
                    np.array([-22.7, 22.05, 0.20]), (-22.7, 20.45)),
    "yoshi":    Obj("yoshi",    "obj_yoshi",    "yoshi_free",    "grab_yoshi",
                    np.array([-21.3, 22.05, 0.20]), (-21.3, 20.45)),
    # green_02 (동, 횡단) — 서쪽에서 접근 (남쪽은 횡단보도 통로로 막힘)
    "android":  Obj("android",  "obj_android",  "android_free",  "grab_android",
                    np.array([13.05, 12.3, 0.20]), (11.0, 12.3)),
    "dino":     Obj("dino",     "obj_dino",     "dino_free",     "grab_dino",
                    np.array([13.05, 13.7, 0.20]), (11.0, 13.7)),
    # green_03 (남)
    "elephant": Obj("elephant", "obj_elephant", "elephant_free", "grab_elephant",
                    np.array([-11.4, -3.95, 0.20]), (-11.4, -5.55)),
    "dog":      Obj("dog",      "obj_dog",      "dog_free",      "grab_dog",
                    np.array([-10.6, -3.95, 0.20]), (-10.6, -5.55)),
}
for o in OBJECTS.values():
    o.body_id = body_id(o.body)
    o.qadr    = jnt_qadr(o.free)
    o.dadr    = jnt_dadr(o.free)
    o.eqid    = eq_id(o.weld)


def reset_objects():
    """키프레임 패딩이 물건 qpos 를 0으로 만드므로 명시적으로 받침 위에 배치."""
    for o in OBJECTS.values():
        data.qpos[o.qadr:o.qadr+3]   = o.start
        data.qpos[o.qadr+3:o.qadr+7] = [1, 0, 0, 0]
        data.qvel[o.dadr:o.dadr+6]   = 0.0
        data.eq_active[o.eqid] = 0


def parse_prompt(text: str) -> Optional[str]:
    t = text.strip().lower()
    for alias, key in NAME_ALIASES.items():
        if alias in t:
            return key
    return None


# ============================================================
# 4. 다리 기구학 + holonomic 보행 (jaemin)
# ============================================================
L1x, L1z, L2 = 0.025, 0.32, 0.3365
HY_MIN, HY_MAX = -0.898845, 2.29511
KN_MIN, KN_MAX = -2.7929, -0.2544
HX_MIN, HX_MAX = -0.785398, 0.785398
HY_HOME, KN_HOME = 1.04, -1.8

def fk(hy, kn):
    c1, s1 = math.cos(hy), math.sin(hy)
    c12, s12 = math.cos(hy+kn), math.sin(hy+kn)
    return c1*L1x - s1*L1z - s12*L2, -s1*L1x - c1*L1z - c12*L2

def leg_ik(fx_t, fz_t, hy=HY_HOME, kn=KN_HOME):
    for _ in range(100):
        fx, fz = fk(hy, kn)
        ex, ez = fx_t-fx, fz_t-fz
        if abs(ex) < 1e-9 and abs(ez) < 1e-9:
            break
        c1, s1 = math.cos(hy), math.sin(hy)
        c12, s12 = math.cos(hy+kn), math.sin(hy+kn)
        J00 = -s1*L1x - c1*L1z - c12*L2; J01 = -c12*L2
        J10 = -c1*L1x + s1*L1z + s12*L2; J11 = s12*L2
        det = J00*J11 - J01*J10
        if abs(det) < 1e-10:
            break
        hy = clamp(hy + 0.5*(J11*ex - J01*ez)/det, HY_MIN, HY_MAX)
        kn = clamp(kn + 0.5*(-J10*ex + J00*ez)/det, KN_MIN, KN_MAX)
    return hy, kn

FX0, FZ0 = fk(HY_HOME, KN_HOME)
L_FOOT_Z = abs(FZ0)

# 보행 파라미터 (큰 맵 → 약간 빠르게)
FREQ, DUTY, STEP_H = 2.4, 0.65, 0.13
SWING_FILTER = 0.55
MAX_VX, MAX_VYAW = 0.75, 1.2

LEGS = {
    "FL": dict(hx=fl_hx, hy=fl_hy, kn=fl_kn, off=0.0, side=+1, hx_pos=+0.298, hy_pos=+0.166),
    "HR": dict(hx=hr_hx, hy=hr_hy, kn=hr_kn, off=0.0, side=-1, hx_pos=-0.298, hy_pos=-0.166),
    "FR": dict(hx=fr_hx, hy=fr_hy, kn=fr_kn, off=0.5, side=-1, hx_pos=+0.298, hy_pos=-0.166),
    "HL": dict(hx=hl_hx, hy=hl_hy, kn=hl_kn, off=0.5, side=+1, hx_pos=-0.298, hy_pos=+0.166),
}
_prev = {n: dict(hx=0.0, hy=HY_HOME, kn=KN_HOME) for n in LEGS}

def foot_traj(phase, vx, vyaw, hip_x, hip_y):
    Sx = (vx - vyaw*hip_y) / FREQ
    Sy = (vyaw*hip_x) / FREQ
    halfx, halfy = Sx/2.0, Sy/2.0
    if phase < DUTY:
        t = phase/DUTY
        dx = halfx - Sx*t; dy = halfy - Sy*t; dz = 0.0
    else:
        t = (phase-DUTY)/(1.0-DUTY)
        dx = -halfx + Sx*t; dy = -halfy + Sy*t; dz = STEP_H*math.sin(math.pi*t)
    return FX0+dx, FZ0+dz, dy

def apply_legs(t_sim, vx, vyaw):
    for name, cfg in LEGS.items():
        phase = (FREQ*t_sim + cfg["off"]) % 1.0
        in_stance = phase < DUTY
        fx_t, fz_t, dy_t = foot_traj(phase, vx, vyaw, cfg["hx_pos"], cfg["hy_pos"])
        hy_t, kn_t = leg_ik(fx_t, fz_t)
        hx_t = clamp(math.asin(clamp(dy_t/L_FOOT_Z, -0.99, 0.99)), HX_MIN, HX_MAX)
        if in_stance:
            data.ctrl[cfg["hx"]] = hx_t
            data.ctrl[cfg["hy"]] = hy_t
            data.ctrl[cfg["kn"]] = kn_t
            _prev[name] = dict(hx=hx_t, hy=hy_t, kn=kn_t)
        else:
            p = _prev[name]
            hx = SWING_FILTER*p["hx"] + (1-SWING_FILTER)*hx_t
            hy = SWING_FILTER*p["hy"] + (1-SWING_FILTER)*hy_t
            kn = SWING_FILTER*p["kn"] + (1-SWING_FILTER)*kn_t
            data.ctrl[cfg["hx"]] = hx; data.ctrl[cfg["hy"]] = hy; data.ctrl[cfg["kn"]] = kn
            _prev[name] = dict(hx=hx, hy=hy, kn=kn)

def hold_legs_stand():
    hy0, kn0 = leg_ik(FX0, FZ0)
    for cfg in LEGS.values():
        data.ctrl[cfg["hx"]] = 0.0
        data.ctrl[cfg["hy"]] = hy0
        data.ctrl[cfg["kn"]] = kn0


# ============================================================
# 5. 팔 자세 / IK / 그리퍼
# ============================================================
ARM_SEED = dict(sh0=0.0, sh1=-1.30, el0=2.20, el1=0.0, wr0=-0.50, wr1=0.0)
ARM_STOW = dict(sh0=0.0, sh1=-3.14, el0=3.06, el1=0.0, wr0=0.0, wr1=0.0)
GRIP_OPEN, GRIP_CLOSE = -1.50, 0.00

ARM_MAX_RATE  = 0.04
GRIP_MAX_RATE = 0.004
EE_TOL, IK_MAX_ITER, IK_STEP, IK_LAMBDA = 8e-3, 80, 0.50, 0.10
ORI_WEIGHT, ORI_TOL = 1.0, 0.10

_arm_ctrl_state = [0.0]*6
_grip_ctrl_state = [0.0]

def set_arm_pose(pose, gripper):
    for i, aid in enumerate(ARM_ACT_IDS):
        data.ctrl[aid] = pose[ARM_JNT[i].replace("arm_", "")]
    data.ctrl[arm_f1x] = gripper

def init_arm_state_from_ctrl():
    for i, aid in enumerate(ARM_ACT_IDS):
        _arm_ctrl_state[i] = float(data.ctrl[aid])
    _grip_ctrl_state[0] = float(data.ctrl[arm_f1x])

def step_arm_toward(target_q, gripper, max_rate=ARM_MAX_RATE):
    for i, aid in enumerate(ARM_ACT_IDS):
        cur = _arm_ctrl_state[i]
        d = clamp(target_q[i]-cur, -max_rate, max_rate)
        lo, hi = ARM_RANGE[i]
        new = clamp(cur+d, lo, hi)
        _arm_ctrl_state[i] = new
        data.ctrl[aid] = new
    gcur = _grip_ctrl_state[0]
    gd = clamp(gripper-gcur, -GRIP_MAX_RATE, GRIP_MAX_RATE)
    _grip_ctrl_state[0] = gcur+gd
    data.ctrl[arm_f1x] = _grip_ctrl_state[0]

def _ee_tip_world(d, bid=EE_BODY_ID):
    pos = d.xpos[bid]
    rot = d.xmat[bid].reshape(3, 3)
    return pos + rot @ EE_LOCAL_OFFSET

def solve_arm_ik(target_world, seed=None, approach_axis=None):
    ik_data.qpos[:] = data.qpos[:]
    ik_data.qvel[:] = 0.0
    seed_vals = list(seed) if seed is not None else [data.ctrl[a] for a in ARM_ACT_IDS]
    for i, qadr in enumerate(ARM_QPOS_ADR):
        lo, hi = ARM_RANGE[i]
        ik_data.qpos[qadr] = clamp(seed_vals[i], lo, hi)
    use_ori = approach_axis is not None
    if use_ori:
        des = np.asarray(approach_axis, float)
        des = des / (np.linalg.norm(des)+1e-12)
    mujoco.mj_forward(ik_model, ik_data)
    jacp = np.zeros((3, ik_model.nv)); jacr = np.zeros((3, ik_model.nv))
    last = float("inf")
    for _ in range(IK_MAX_ITER):
        mujoco.mj_forward(ik_model, ik_data)
        ee = _ee_tip_world(ik_data)
        perr = target_world - ee
        last = float(np.linalg.norm(perr))
        mujoco.mj_jac(ik_model, ik_data, jacp, jacr, ee, EE_BODY_ID)
        if use_ori:
            cur = ik_data.xmat[EE_BODY_ID].reshape(3, 3)[:, 0]
            oerr = np.cross(cur, des); omag = float(np.linalg.norm(oerr))
            if last < EE_TOL and omag < ORI_TOL:
                break
            err6 = np.concatenate([perr, ORI_WEIGHT*oerr])
            J = np.vstack([jacp[:, ARM_DOF_ADR], jacr[:, ARM_DOF_ADR]])
            dq = J.T @ np.linalg.solve(J@J.T + (IK_LAMBDA**2)*np.eye(6), err6)
        else:
            if last < EE_TOL:
                break
            J = jacp[:, ARM_DOF_ADR]
            dq = J.T @ np.linalg.solve(J@J.T + (IK_LAMBDA**2)*np.eye(3), perr)
        for i, qadr in enumerate(ARM_QPOS_ADR):
            lo, hi = ARM_RANGE[i]
            ik_data.qpos[qadr] = clamp(ik_data.qpos[qadr] + IK_STEP*dq[i], lo, hi)
    return [float(ik_data.qpos[q]) for q in ARM_QPOS_ADR], last

def arm_seed_list():
    return [ARM_SEED[k] for k in ["sh0","sh1","el0","el1","wr0","wr1"]]

def arm_stow_list():
    return [ARM_STOW[k] for k in ["sh0","sh1","el0","el1","wr0","wr1"]]


# ============================================================
# 6. weld 부착 / 해제
# ============================================================
def attach(obj: Obj) -> bool:
    """그리퍼-물건 weld 활성화 (현재 상대 자세 유지).
    그리퍼 끝이 물건에서 충분히 가까울 때만 부착(멀리서 부착 시 시뮬 폭발 방지).
    """
    ee = _ee_tip_world(data)
    op = data.xpos[obj.body_id]
    if float(np.linalg.norm(ee - op)) > 0.16:
        return False
    b1, b2 = EE_BODY_ID, obj.body_id
    p1, q1 = data.xpos[b1], data.xquat[b1]
    p2, q2 = data.xpos[b2], data.xquat[b2]
    negq1 = np.zeros(4); mujoco.mju_negQuat(negq1, q1)
    relpos = np.zeros(3); mujoco.mju_rotVecQuat(relpos, p2-p1, negq1)
    relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, negq1, q2)
    model.eq_data[obj.eqid, 0:3]  = 0.0
    model.eq_data[obj.eqid, 3:6]  = relpos
    model.eq_data[obj.eqid, 6:10] = relquat
    model.eq_data[obj.eqid, 10]   = 1.0
    data.eq_active[obj.eqid] = 1
    return True

def detach(obj: Obj):
    data.eq_active[obj.eqid] = 0


# ============================================================
# 7. 비전 : 로봇 시점 카메라에서 신호등 초록 검출
# ============================================================
TL_BOTTOM = np.array([1.8, 2.0, 2.0])    # 남쪽 신호등
TL_TOP    = np.array([-1.8, 8.0, 2.0])   # 북쪽 신호등

class Vision:
    def __init__(self, w=160, h=120):
        self.renderer = mujoco.Renderer(model, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.last_ratio = 0.0
        self.last_img = np.zeros((h, w, 3), dtype=np.uint8)
        self.last_gp = self.last_rp = 0

    def detect_green(self, robot_x, robot_y, light=None):
        """로봇에서 가까운 신호등을 바라본 영상에서 초록 픽셀 비율 검출."""
        if light is None:   # 가까운 신호등 선택 (남/북)
            db = math.hypot(robot_x-TL_BOTTOM[0], robot_y-TL_BOTTOM[1])
            dt = math.hypot(robot_x-TL_TOP[0], robot_y-TL_TOP[1])
            light = TL_BOTTOM if db <= dt else TL_TOP
        dx, dy = robot_x - light[0], robot_y - light[1]
        dist_h = math.hypot(dx, dy)
        self.cam.lookat[:] = light
        self.cam.distance = max(2.0, dist_h)
        self.cam.azimuth = math.degrees(math.atan2(dy, dx))
        self.cam.elevation = -8.0
        self.renderer.update_scene(data, self.cam)
        rgb = self.renderer.render()
        self.last_img = rgb.copy()        # 녹화용 보관
        img = rgb.astype(np.int16)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        # 초록 픽셀 vs 빨강 픽셀 비교 (운반 물건이 일부 가려도 강건)
        gp = int(((g > 140) & (r < 100) & (b < 100)).sum())
        rp = int(((r > 140) & (g < 100) & (b < 100)).sum())
        self.last_ratio = gp / max(1, g.size)
        self.last_gp, self.last_rp = gp, rp
        return gp > rp and gp > 10        # 밝은 초록이 빨강보다 우세하면 green(ON)

    def shot(self, camera):
        self.renderer.update_scene(data, camera=camera)
        return self.renderer.render().copy()


# ============================================================
# 8. 경로 추종 (pure-pursuit, holonomic 보행 명령 생성)
# ============================================================
WP_REACH, GOAL_REACH = 0.6, 0.7
KP_YAW, HEADING_SLOW = 1.8, 0.6

@dataclass
class PathFollower:
    wps: List[Tuple[float, float]] = field(default_factory=list)
    idx: int = 0
    def reset(self, wps):
        self.wps = list(wps); self.idx = 0
    def done(self):
        return self.idx >= len(self.wps)
    def step(self, x, y, yaw, goal_xy=None):
        if self.done():
            return 0.0, 0.0, True
        while self.idx < len(self.wps)-1:
            wx, wy = self.wps[self.idx]
            if math.hypot(wx-x, wy-y) < WP_REACH:
                self.idx += 1
            else:
                break
        wx, wy = self.wps[self.idx]
        dx, dy = wx-x, wy-y
        dist = math.hypot(dx, dy)
        yaw_err = wrap_angle(math.atan2(dy, dx) - yaw)
        if goal_xy is not None and math.hypot(goal_xy[0]-x, goal_xy[1]-y) < GOAL_REACH:
            return 0.0, 0.0, True
        if abs(yaw_err) > HEADING_SLOW:
            vx = 0.12
            vyaw = clamp(KP_YAW*yaw_err, -MAX_VYAW, MAX_VYAW)
        else:
            vx = clamp(MAX_VX*(1-abs(yaw_err)/HEADING_SLOW*0.5), 0.0, MAX_VX)
            # 헤딩 오차가 작으면 회전 명령 제거 → 직진 시 좌우 흔들림 억제
            vyaw = 0.0 if abs(yaw_err) < 0.05 else clamp(KP_YAW*yaw_err, -MAX_VYAW, MAX_VYAW)
        if self.idx == len(self.wps)-1 and dist < 1.0:
            vx *= max(dist/1.0, 0.35)
        return vx, vyaw, False


class CrosswalkGate:
    """양방향 횡단보도 게이트: 도로 진입 전(남/북 양쪽) 정지하고
    비전이 '신선한 초록'(빨강을 본 뒤 초록)일 때만 건넌다."""
    ROAD_S, ROAD_N = 2.5, 7.5
    def __init__(self):
        self.state = "idle"; self.saw_red = False
    def update(self, bx, by, vision_green):
        if abs(bx) > 1.6:
            self.state = "idle"; self.saw_red = False
            return False
        if self.state == "crossing":
            if by < 0.6 or by > 9.4:
                self.state = "idle"; self.saw_red = False
            return False
        south_zone = 0.8 < by <= 2.7      # 도로 남쪽 정지구역
        north_zone = 7.3 <= by < 9.2      # 도로 북쪽 정지구역
        if south_zone or north_zone:
            if not vision_green:
                self.saw_red = True
                self.state = "waiting"
                return True               # 빨강 → 정지 대기
            if self.saw_red:
                self.state = "crossing"   # 신선한 초록 → 출발
                print("[횡단보도] 비전 초록 인지 → 횡단 시작")
                return False
            self.state = "waiting"        # 초록에 도착 → 한 번 빨강 기다림
            return True
        return False


def plan(start_xy, goal_xy):
    raw = pathfinding.astar(start_xy, goal_xy)
    if not raw:
        return []
    wps = pathfinding.smooth_path(raw, min_dist=1.2)
    wps[-1] = goal_xy
    # 도로(y≈5) 횡단 시 횡단보도 중앙선(x=0)을 직진 통과하도록 경유점 강제
    if (start_xy[1] - 5.0) * (goal_xy[1] - 5.0) < 0:
        if start_xy[1] > goal_xy[1]:                       # 북 → 남
            before = [w for w in wps if w[1] > 9.0]
            after  = [w for w in wps if w[1] < 1.0]
            wps = before + [(0.0, 8.8), (0.0, 1.0)] + after
        else:                                             # 남 → 북
            before = [w for w in wps if w[1] < 1.0]
            after  = [w for w in wps if w[1] > 9.0]
            wps = before + [(0.0, 1.0), (0.0, 8.8)] + after
        if wps[-1] != goal_xy:
            wps.append(goal_xy)
    return wps


# ============================================================
# 9. FSM
# ============================================================
class S:
    GO = "GO_TO_OBJECT"
    APPROACH = "APPROACH"
    PRE = "PRE_GRASP"
    DESCEND = "DESCEND"
    CLOSE = "CLOSE"
    LIFT = "LIFT"
    RETURN = "RETURN"
    PLACE_DOWN = "PLACE_DOWN"
    RELEASE = "RELEASE"
    DONE = "DONE"

FINAL_XY = (0.0, -19.0)         # 최종 내려놓는 위치 (red_mark_02)
APPROACH_DIST = 0.62
APPROACH_REACH_MAX = 0.74
APPROACH_VX_MIN = 0.50    # 저속 보행 게이트 회피 (0.3 에선 제자리걸음)
PRE_GRASP_DZ = 0.18
LIFT_HEIGHT = 0.55


def stabilize(viewer, n=4000):
    hy0, kn0 = leg_ik(FX0, FZ0)
    for _ in range(n):
        for cfg in LEGS.values():
            data.ctrl[cfg["hx"]] = 0.0
            data.ctrl[cfg["hy"]] = hy0
            data.ctrl[cfg["kn"]] = kn0
        set_arm_pose(ARM_STOW, GRIP_OPEN)
        mujoco.mj_step(model, data)
    for n_ in LEGS:
        _prev[n_] = dict(hx=0.0, hy=hy0, kn=kn0)
    init_arm_state_from_ctrl()


def main(target_key="mario", headless=False, max_sim_time=240.0, show_cam=False,
         record=False, record_dir="frames", speed=1.0):
    obj = OBJECTS[target_key]
    print("="*64)
    print(f"  통합 시나리오 :  '{target_key}' 잡아오기"
          + ("  [HEADLESS]" if headless else ""))
    print("="*64)

    traffic.init(model, data)
    cars.init(model, data)

    import contextlib
    viewer_cm = (contextlib.nullcontext(None) if headless
                 else mujoco.viewer.launch_passive(model, data))
    with viewer_cm as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        reset_objects()
        set_arm_pose(ARM_STOW, GRIP_OPEN)
        mujoco.mj_forward(model, data)
        print("[init] 안정화 ...")
        stabilize(viewer)
        if viewer:
            viewer.sync()

        vision = Vision()
        gate = CrosswalkGate()

        # ── 로봇 카메라 별창 표시 (--cam) ──
        cam_proc = cam_queue = cam_rend = None
        if show_cam and not headless:
            try:
                from multiprocessing import Process, Queue as MPQueue
                import _cam_view
                cam_rend = mujoco.Renderer(model, height=240, width=320)
                cam_queue = MPQueue(maxsize=1)
                cam_proc = Process(target=_cam_view.run, args=(cam_queue,), daemon=True)
                cam_proc.start()
                print("[cam] 로봇 카메라 창 열림 (body/arm/traffic-light)")
            except Exception as e:
                print(f"[cam] 카메라 창 실패: {e}")
                cam_proc = None

        # ── 녹화 설정 (추적 카메라 + 비전 화면 합성 저장) ──
        rec = reccam = None
        if record:
            from PIL import Image, ImageDraw
            os.makedirs(record_dir, exist_ok=True)
            for f in os.listdir(record_dir):
                if f.endswith(".png"):
                    os.remove(os.path.join(record_dir, f))
            rec = mujoco.Renderer(model, height=480, width=640)
            reccam = mujoco.MjvCamera()
            reccam.type = mujoco.mjtCamera.mjCAMERA_FREE
            reccam.azimuth, reccam.elevation, reccam.distance = 135.0, -28.0, 12.0
            _frame_n = [0]

            def save_frame(tag, t_, st, tl_, vis_):
                reccam.lookat[:] = data.xpos[BODY_ID]
                rec.update_scene(data, reccam)
                scene = rec.render()
                canvas = Image.fromarray(scene).convert("RGB")
                vimg = Image.fromarray(vision.last_img).resize((213, 160))
                canvas.paste(vimg, (640 - 213 - 6, 6))
                d = ImageDraw.Draw(canvas)
                d.rectangle([0, 0, 360, 60], fill=(0, 0, 0))
                d.text((6, 4),  f"t={t_:6.1f}  {st}", fill=(255, 255, 255))
                d.text((6, 22), f"light(truth)={tl_}  vision={vis_}", fill=(150, 255, 150))
                d.text((6, 40), f"obj={target_key}  gp={vision.last_gp} rp={vision.last_rp}",
                       fill=(255, 220, 120))
                d.text((640 - 213 - 6, 168), "ROBOT VISION (traffic light)", fill=(200, 200, 255))
                canvas.save(os.path.join(record_dir, f"{_frame_n[0]:04d}_{tag}.png"))
                _frame_n[0] += 1

        # 경로 계획: 시작 → 물건 접근점
        start_xy = (float(data.qpos[BASE]), float(data.qpos[BASE+1]))
        follower = PathFollower()
        follower.reset(plan(start_xy, obj.stop))
        print(f"[plan] start={start_xy} → stop={obj.stop}  wps={len(follower.wps)}")
        if not follower.wps:
            print("[plan] ⚠ 경로를 찾지 못했습니다 (staging 점이 막혀있을 수 있음)")

        state, state_t0 = S.GO, 0.0
        prev_state = None
        ik_target_q = arm_stow_list()
        ik_gripper = GRIP_OPEN
        ik_axis = None
        vis_state = "red"
        vis_green_cache = False

        dt = model.opt.timestep
        t, step_i = 0.0, 0
        RAMP = 1.5
        log_int = int(1.0/dt)
        vision_stride = max(1, int(0.1/dt))   # 신호등 인지는 0.1초마다만 렌더(배속 유지)
        # 배속: speed 배 빠르게. 고배속에선 렌더 빈도를 줄여 시뮬이 따라가게.
        render_stride = 1 if speed <= 4 else max(1, int(speed / 2))
        if not headless and speed != 1.0:
            print(f"[배속] {speed}x  (렌더 1/{render_stride} 스텝)")

        while (viewer.is_running() if viewer
               else (t < max_sim_time and state != S.DONE)):
            bx, by = data.qpos[BASE], data.qpos[BASE+1]
            bz = data.qpos[BASE+2]
            _, _, yaw = quat_to_rpy(data.qpos[BASE+3:BASE+7])
            ox, oy, oz = data.qpos[obj.qadr:obj.qadr+3]

            tl_truth = traffic.update(t)     # 신호등 색 갱신(지면 진실)
            vx_cmd, vyaw_cmd = 0.0, 0.0
            arm_rate = ARM_MAX_RATE

            # ───── 내비게이션 단계 (GO / RETURN) ─────
            if state in (S.GO, S.RETURN):
                goal = obj.stop if state == S.GO else FINAL_XY
                vx_cmd, vyaw_cmd, arrived = follower.step(bx, by, yaw, goal_xy=goal)
                if state == S.GO:
                    ik_target_q = arm_stow_list()   # 갈 때는 팔 접음
                    ik_gripper = GRIP_OPEN
                else:
                    ik_gripper = GRIP_CLOSE          # 올 때는 LIFT 자세 유지(물건 운반)
                ik_axis = None

                # 횡단보도(양방향): 비전으로 신호등 초록 인지 → 통과 결정.
                # 비전 렌더는 무거우므로 0.1초마다만 갱신하고 결과 캐시(배속 유지).
                if abs(bx) < 1.6 and 0.5 < by < 9.5:
                    if step_i % vision_stride == 0:
                        vis_green_cache = vision.detect_green(bx, by)
                    vg = vis_green_cache
                    vis_state = "green" if vg else "red"
                    if gate.update(bx, by, vg):
                        vx_cmd, vyaw_cmd = 0.0, 0.0
                else:
                    gate.update(bx, by, False)

                if arrived:
                    if state == S.GO:
                        print(f"[FSM] 도착 → APPROACH  pos=({bx:+.2f},{by:+.2f})")
                        state, state_t0 = S.APPROACH, t
                        init_arm_state_from_ctrl()
                    else:
                        print(f"[FSM] 복귀완료 → PLACE_DOWN")
                        state, state_t0 = S.PLACE_DOWN, t
                        init_arm_state_from_ctrl()

            elif state == S.APPROACH:
                dx, dy = ox - bx, oy - by
                yaw_err = wrap_angle(math.atan2(dy, dx) - yaw)
                dist = math.hypot(dx, dy)
                err = dist - APPROACH_DIST
                vyaw_cmd = clamp(2.0*yaw_err, -MAX_VYAW, MAX_VYAW)
                if abs(yaw_err) > 0.5:
                    vx_cmd = 0.0
                elif dist > APPROACH_DIST:
                    vx_cmd = clamp(0.9*err, APPROACH_VX_MIN, MAX_VX)
                else:
                    vx_cmd = clamp(0.6*err, -0.15, 0.10)
                ik_target_q = arm_stow_list()    # 접근 중 팔 접음(물건 밀침 방지)
                ik_gripper = GRIP_OPEN
                aligned = (0.40 < dist < APPROACH_REACH_MAX and abs(yaw_err) < 0.15)
                if aligned:
                    print(f"[FSM] APPROACH → PRE_GRASP dist={dist:.2f} yaw_err={math.degrees(yaw_err):+.0f}")
                    state, state_t0 = S.PRE, t
                    init_arm_state_from_ctrl()
                elif (t-state_t0) > 12.0:
                    if dist < 1.0:        # 거의 도달 → 진행
                        state, state_t0 = S.PRE, t
                        init_arm_state_from_ctrl()
                    else:                 # 멀리서 못 닿음 → 파지 금지(가짜 파지 방지)
                        print(f"[FSM] APPROACH 실패: 물건까지 {dist:.1f}m → 중단")
                        state, state_t0 = S.DONE, t

            elif state == S.PRE:
                tgt = np.array([ox, oy, oz + PRE_GRASP_DZ])
                ik_target_q, _ = solve_arm_ik(tgt, seed=arm_seed_list(), approach_axis=GRASP_AXIS)
                ik_gripper, ik_axis = GRIP_OPEN, GRASP_AXIS
                if float(np.linalg.norm(_ee_tip_world(data)-tgt)) < 0.04 or (t-state_t0) > 3.0:
                    print("[FSM] PRE_GRASP → DESCEND")
                    state, state_t0 = S.DESCEND, t

            elif state == S.DESCEND:
                # 그리퍼를 물건 위로 내린 뒤, 물건을 그리퍼 끝으로 스냅하고 weld(안정적 파지)
                tgt = np.array([ox, oy, oz + 0.02])
                ik_target_q, _ = solve_arm_ik(tgt, seed=ik_target_q, approach_axis=GRASP_AXIS)
                ik_gripper = GRIP_OPEN
                arm_rate = 0.02
                ee = _ee_tip_world(data)
                ee_d = float(np.linalg.norm(ee - data.xpos[obj.body_id]))
                if ee_d < 0.16 or (t-state_t0) > 2.5:
                    # 스냅 그랩: 물건을 그리퍼 끝으로 이동 후 weld (상대거리 ~0 → 안정)
                    data.qpos[obj.qadr:obj.qadr+3]   = ee
                    data.qpos[obj.qadr+3:obj.qadr+7] = [1, 0, 0, 0]
                    data.qvel[obj.dadr:obj.dadr+6]   = 0.0
                    mujoco.mj_forward(model, data)
                    attach(obj)
                    print(f"[FSM] DESCEND: 스냅 grab + weld (ee_d={ee_d:.3f})")
                    state, state_t0 = S.CLOSE, t

            elif state == S.CLOSE:
                # 부착 후 그리퍼 닫는 모션(시각용) 잠깐
                ik_gripper = GRIP_CLOSE
                if (t-state_t0) > 1.0:
                    state, state_t0 = S.LIFT, t

            elif state == S.LIFT:
                tgt = np.array([bx + 0.55*math.cos(yaw), by + 0.55*math.sin(yaw), LIFT_HEIGHT])
                ik_target_q, _ = solve_arm_ik(tgt, seed=ik_target_q, approach_axis=GRASP_AXIS)
                ik_gripper = GRIP_CLOSE
                if (t-state_t0) > 2.5:
                    print(f"[FSM] LIFT → RETURN  obj_z={oz:.2f}")
                    follower.reset(plan((bx, by), FINAL_XY))
                    state, state_t0 = S.RETURN, t

            elif state == S.PLACE_DOWN:
                # 들고 온 물건을 바닥 가까이 내려놓을 위치로 팔 하강
                tgt = np.array([bx + 0.55*math.cos(yaw), by + 0.55*math.sin(yaw), 0.10])
                ik_target_q, _ = solve_arm_ik(tgt, seed=ik_target_q, approach_axis=GRASP_AXIS)
                ik_gripper = GRIP_CLOSE
                if (t-state_t0) > 3.0:
                    state, state_t0 = S.RELEASE, t

            elif state == S.RELEASE:
                detach(obj)
                ik_gripper = GRIP_OPEN
                if (t-state_t0) > 1.5:
                    fox, foy, foz = data.qpos[obj.qadr:obj.qadr+3]
                    print(f"[FSM] RELEASE → DONE  최종 물건 위치=({fox:+.2f},{foy:+.2f},{foz:.2f})")
                    state, state_t0 = S.DONE, t
            else:
                ik_gripper = GRIP_CLOSE

            # ───── 보행 / 팔 / 차량 ─────
            if abs(vx_cmd) < 1e-3 and abs(vyaw_cmd) < 1e-3:
                hold_legs_stand()
            else:
                ramp = min(t/RAMP, 1.0)
                apply_legs(t, vx_cmd*ramp, vyaw_cmd*ramp)
            step_arm_toward(ik_target_q, ik_gripper, max_rate=arm_rate)
            cars.apply(tl_truth)
            mujoco.mj_step(model, data)
            if viewer and step_i % render_stride == 0:
                viewer.sync()

            # 로봇 카메라 창 갱신 (0.1초마다, 큐가 비었을 때만)
            if cam_proc is not None and step_i % vision_stride == 0 and not cam_queue.full():
                try:
                    cam_rend.update_scene(data, camera="body_cam")
                    bimg = cam_rend.render().copy()
                    cam_rend.update_scene(data, camera="arm_cam")
                    aimg = cam_rend.render().copy()
                    from PIL import Image as _PImg
                    limg = np.asarray(_PImg.fromarray(vision.last_img).resize((320, 240)))
                    cam_queue.put_nowait((bimg, aimg, limg))
                except Exception:
                    pass

            if record and (state != prev_state or step_i % int(3.0/dt) == 0):
                save_frame(state, t, state, tl_truth, vis_state)
            prev_state = state

            if step_i % log_int == 0:
                print(f"[t={t:6.1f}] {state:<11} pos=({bx:+.1f},{by:+.1f}) "
                      f"yaw={math.degrees(yaw):+5.0f} v=({vx_cmd:+.2f},{vyaw_cmd:+.2f}) "
                      f"tl={tl_truth}/{vis_state} obj=({ox:+.1f},{oy:+.1f},{oz:.2f})")

            t += dt; step_i += 1
            if viewer:
                time.sleep(max(0.0, dt/speed - 0.0001))

        if cam_proc is not None:
            try:
                cam_queue.put_nowait(None)
            except Exception:
                pass

        if headless:
            fox, foy, foz = data.qpos[obj.qadr:obj.qadr+3]
            ok = (state == S.DONE and math.hypot(fox-FINAL_XY[0], foy-FINAL_XY[1]) < 2.0)
            print("="*64)
            print(f"[RESULT] state={state} t={t:.1f}s  obj=({fox:+.2f},{foy:+.2f},{foz:.2f})")
            print(f"[RESULT] {'SUCCESS' if ok else 'FAIL'}")
            print("="*64)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = sys.argv[1:]
    headless = "--headless" in args
    record = "--record" in args
    show_cam = "--cam" in args
    key = "mario"
    speed = 1.0
    obj_given = False
    for a in args:
        if a.startswith("--obj="):
            key = a.split("=", 1)[1]; obj_given = True
        elif a.startswith("--speed="):
            speed = float(a.split("=", 1)[1])
        elif a == "--fast":
            speed = 5.0
    if not headless and not obj_given:
        # 대화형: 프롬프트 입력 (--obj 미지정 시에만)
        try:
            txt = input("무엇을 잡아올까요? (예: '사과 잡아와') : ")
            k = parse_prompt(txt)
            if k:
                key = k
            else:
                print(f"인식 실패 → 기본값 '{key}'")
        except EOFError:
            pass
    print(f"[목표 물건] {key}  (speed={speed}x{'  +cam' if show_cam else ''})")
    main(target_key=key, headless=headless or record, record=record,
         speed=speed, show_cam=show_cam)
