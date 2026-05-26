"""
main_jaemin.py  (v2: IK-based grasping)
================================================================
Spot (Boston Dynamics) 4족보행 + A* 패스파인딩 + 팔 IK 기반 그래스핑

파이프라인 (시스템 다이어그램 기반)
    1. Occupancy grid + A*  → 최단 경로 (장애물 인플레이션 포함)
    2. Pure-pursuit          → vx, vyaw 명령
    3. Trot 보행             → 4개 다리 PD 제어 (yaw, hx-roll 보정)
    4. APPROACH              → 큐브를 본체 정면 d0(=0.6m) 에 정확히 위치
    5. Arm IK (Damped LS)    → 그리퍼 끝을 큐브 좌표 (월드) 로 수렴
       PRE_GRASP → DESCEND → CLOSE(grip) → LIFT → HOLD
    6. DONE                  → 큐브가 0.30m 이상 들렸으면 성공

v1 대비 주요 변경
    * Goal 위치를 큐브 정면 1.0 m 앞으로 (NAV 끝나면 큐브를 바라봄)
    * APPROACH 도달 조건 완화 + 5 s timeout
    * Keyframe 보간 → MuJoCo Jacobian-based 6-DoF arm IK
    * 별도의 ik_data 모델로 IK 풀이 → 실제 시뮬레이션과 분리
================================================================
"""

import time
import math
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# 0. 모델 로드 (시뮬 + IK 풀이용 사본)
# ============================================================
SCENE_XML = "scene_jaemin.xml"

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data  = mujoco.MjData(model)

# IK 전용 모델/데이터 (실제 sim 과 분리, qpos 만 임시로 빌려 씀)
ik_model = mujoco.MjModel.from_xml_path(SCENE_XML)
ik_data  = mujoco.MjData(ik_model)


# ============================================================
# 1. 유틸리티
# ============================================================
def get_act(name: str) -> int:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid == -1:
        raise ValueError(f"Actuator not found: {name}")
    return aid


def get_jnt_qposadr(name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid == -1:
        raise ValueError(f"Joint not found: {name}")
    return model.jnt_qposadr[jid]


def get_jnt_dofadr(name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid == -1:
        raise ValueError(f"Joint not found: {name}")
    return model.jnt_dofadr[jid]


def get_jnt_range(name: str) -> Tuple[float, float]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = model.jnt_range[jid]
    return float(lo), float(hi)


def get_body_id(name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        raise ValueError(f"Body not found: {name}")
    return bid


def find_free_qpos_body() -> int:
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[j]
    raise RuntimeError("Free joint not found")


def quat_to_rpy(q):
    w, x, y, z = q
    roll  = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw   = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


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

BASE        = find_free_qpos_body()
TARGET_QPOS = get_jnt_qposadr("target_free")
TARGET_DOF  = get_jnt_dofadr("target_free")
BODY_ID     = get_body_id("body")
TARGET_ID   = get_body_id("target_cube")
TARGET_START = np.array([5.5, 3.5, 0.05], dtype=float)

# 팔 관절 정보
ARM_JNT_NAMES = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1"]
ARM_ACT_IDS   = [arm_sh0,   arm_sh1,   arm_el0,   arm_el1,   arm_wr0,   arm_wr1  ]
ARM_QPOS_ADR  = [get_jnt_qposadr(n) for n in ARM_JNT_NAMES]
ARM_DOF_ADR   = [get_jnt_dofadr(n)  for n in ARM_JNT_NAMES]
ARM_RANGE     = [get_jnt_range(n)   for n in ARM_JNT_NAMES]

# End-effector body : 그리퍼 손가락 body
EE_BODY_ID    = get_body_id("arm_link_fngr")
# 그리퍼 끝(파지점) - fngr body 좌표계에서 손가락 앞쪽 0.10m, z 0.0
EE_LOCAL_OFFSET = np.array([0.10, 0.0, 0.0])


def reset_target_cube():
    """Spot keyframe reset 이 target_free qpos 를 0으로 덮어써서 큐브를 원래 위치로 되돌린다."""
    data.qpos[TARGET_QPOS + 0 : TARGET_QPOS + 3] = TARGET_START
    data.qpos[TARGET_QPOS + 3 : TARGET_QPOS + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[TARGET_DOF : TARGET_DOF + 6] = 0.0


# ============================================================
# 3. 다리 기구학 (main.py 그대로)
# ============================================================
L1x = 0.025
L1z = 0.32
L2  = 0.3365

HY_MIN, HY_MAX = -0.898845, 2.29511
KN_MIN, KN_MAX = -2.7929,  -0.2544
HX_MIN, HX_MAX = -0.785398, 0.785398

HY_HOME = 1.04
KN_HOME = -1.8


def fk(hy, kn):
    c1, s1   = math.cos(hy),       math.sin(hy)
    c12, s12 = math.cos(hy + kn),  math.sin(hy + kn)
    return c1 * L1x - s1 * L1z - s12 * L2, -s1 * L1x - c1 * L1z - c12 * L2


def leg_ik(fx_t, fz_t, hy=HY_HOME, kn=KN_HOME):
    for _ in range(100):
        fx, fz = fk(hy, kn)
        ex, ez = fx_t - fx, fz_t - fz
        if abs(ex) < 1e-9 and abs(ez) < 1e-9:
            break
        c1, s1   = math.cos(hy),       math.sin(hy)
        c12, s12 = math.cos(hy + kn),  math.sin(hy + kn)
        J00 = -s1 * L1x - c1 * L1z - c12 * L2
        J01 = -c12 * L2
        J10 = -c1 * L1x + s1 * L1z + s12 * L2
        J11 =  s12 * L2
        det = J00 * J11 - J01 * J10
        if abs(det) < 1e-10:
            break
        hy = clamp(hy + 0.5 * ( J11 * ex - J01 * ez) / det, HY_MIN, HY_MAX)
        kn = clamp(kn + 0.5 * (-J10 * ex + J00 * ez) / det, KN_MIN, KN_MAX)
    return hy, kn


FX0, FZ0 = fk(HY_HOME, KN_HOME)


# ============================================================
# 4. 보행 파라미터
# ============================================================
FREQ          = 2.0
DUTY          = 0.65
STEP_H        = 0.10
SWING_FILTER  = 0.55

MAX_VX        = 0.55
MAX_VYAW      = 1.2

HX_GAIN_YAW   = 0.05


LEGS = {
    "FL": dict(hx=fl_hx, hy=fl_hy, kn=fl_kn, off=0.0, side=+1, hx_pos=+0.298, hy_pos=+0.166),
    "HR": dict(hx=hr_hx, hy=hr_hy, kn=hr_kn, off=0.0, side=-1, hx_pos=-0.298, hy_pos=-0.166),
    "FR": dict(hx=fr_hx, hy=fr_hy, kn=fr_kn, off=0.5, side=-1, hx_pos=+0.298, hy_pos=-0.166),
    "HL": dict(hx=hl_hx, hy=hl_hy, kn=hl_kn, off=0.5, side=+1, hx_pos=-0.298, hy_pos=+0.166),
}

_prev = {n: dict(hx=0.0, hy=HY_HOME, kn=KN_HOME) for n in LEGS}


def foot_traj(phase, vx, vyaw, hip_y):
    stride = vx / FREQ
    yaw_d  = -vyaw * hip_y / FREQ      # 좌회전 시 우측 다리 stride↑
    S      = stride + yaw_d
    half   = S / 2.0

    if phase < DUTY:
        t  = phase / DUTY
        dx = half - S * t
        dz = 0.0
    else:
        t  = (phase - DUTY) / (1.0 - DUTY)
        dx = -half + S * t
        dz = STEP_H * math.sin(math.pi * t)

    return FX0 + dx, FZ0 + dz


def apply_legs(t_sim, vx, vyaw):
    for name, cfg in LEGS.items():
        phase     = (FREQ * t_sim + cfg["off"]) % 1.0
        in_stance = phase < DUTY

        fx_t, fz_t = foot_traj(phase, vx, vyaw, cfg["hy_pos"])
        hy_t, kn_t = leg_ik(fx_t, fz_t)
        hx_t       = clamp(HX_GAIN_YAW * vyaw * cfg["side"], HX_MIN, HX_MAX)

        if in_stance:
            data.ctrl[cfg["hx"]] = hx_t
            data.ctrl[cfg["hy"]] = hy_t
            data.ctrl[cfg["kn"]] = kn_t
            _prev[name] = dict(hx=hx_t, hy=hy_t, kn=kn_t)
        else:
            p  = _prev[name]
            hx = SWING_FILTER * p["hx"] + (1 - SWING_FILTER) * hx_t
            hy = SWING_FILTER * p["hy"] + (1 - SWING_FILTER) * hy_t
            kn = SWING_FILTER * p["kn"] + (1 - SWING_FILTER) * kn_t
            data.ctrl[cfg["hx"]] = hx
            data.ctrl[cfg["hy"]] = hy
            data.ctrl[cfg["kn"]] = kn
            _prev[name] = dict(hx=hx, hy=hy, kn=kn)


def hold_legs_stand():
    """정지 자세 (홈포즈)"""
    hy0, kn0 = leg_ik(FX0, FZ0)
    for cfg in LEGS.values():
        data.ctrl[cfg["hx"]] = 0.0
        data.ctrl[cfg["hy"]] = hy0
        data.ctrl[cfg["kn"]] = kn0


# ============================================================
# 5. 팔 자세 (기본 / 그리퍼)
# ============================================================
# 시드 자세 - IK 가 풀기 좋은 초기값
ARM_SEED = dict(
    sh0= 0.00, sh1=-1.30, el0=2.20, el1=0.00,
    wr0=-0.50, wr1= 0.00,
)
ARM_STOW = dict(
    sh0= 0.00, sh1=-3.14, el0=3.06, el1=0.00,
    wr0= 0.00, wr1= 0.00,
)

GRIP_OPEN  = -1.50    # f1x 열림 (큐브 통과 가능)
GRIP_CLOSE =  0.00    # f1x 닫힘


def set_arm_joints(joint_targets: List[float], gripper: float):
    """joint_targets = [sh0, sh1, el0, el1, wr0, wr1]"""
    for i, aid in enumerate(ARM_ACT_IDS):
        data.ctrl[aid] = joint_targets[i]
    data.ctrl[arm_f1x] = gripper


def set_arm_pose(pose: dict, gripper: float):
    targets = [pose[k] for k in ["sh0", "sh1", "el0", "el1", "wr0", "wr1"]]
    set_arm_joints(targets, gripper)


# ============================================================
# 6. Jacobian 기반 팔 IK  (Damped Least Squares)
# ============================================================
EE_TOL          = 8e-3     # 수렴 임계 (m, 8mm)
IK_MAX_ITER     = 80       # 보통 10~20 iter 면 수렴
IK_STEP         = 0.50     # 각 IK 스텝의 update gain
IK_LAMBDA       = 0.10     # damping λ  (λ²=0.01)


def _ee_tip_world(d, body_id=EE_BODY_ID):
    """그리퍼 끝점의 world 좌표 (EE_LOCAL_OFFSET 적용)"""
    pos = d.xpos[body_id]
    rot = d.xmat[body_id].reshape(3, 3)
    return pos + rot @ EE_LOCAL_OFFSET


def solve_arm_ik(target_world: np.ndarray,
                 seed: Optional[List[float]] = None,
                 verbose: bool = False) -> Tuple[List[float], float]:
    """
    그리퍼 끝(EE)을 target_world (월드 좌표) 로 보내는 6-DoF 팔 IK.
    실제 시뮬레이션과 분리된 ik_data 위에서 풀이.
    반환: ([sh0, sh1, el0, el1, wr0, wr1],  최종 error norm)
    """
    # 현재 본체 자세는 그대로 두고, 팔 qpos 만 seed 또는 현재 ctrl 로 셋팅
    ik_data.qpos[:] = data.qpos[:]
    ik_data.qvel[:] = 0.0

    if seed is None:
        seed_vals = [data.ctrl[a] for a in ARM_ACT_IDS]
    else:
        seed_vals = list(seed)
    for i, qadr in enumerate(ARM_QPOS_ADR):
        lo, hi = ARM_RANGE[i]
        ik_data.qpos[qadr] = clamp(seed_vals[i], lo, hi)

    # NOTE: mj_jac 은 forward dynamics 의 부산물 (cdof, com 등) 을 필요로 하므로
    #       단순 mj_kinematics 만으로는 dq=0 이 되어버린다 → mj_forward 사용
    mujoco.mj_forward(ik_model, ik_data)

    jacp = np.zeros((3, ik_model.nv))
    jacr = np.zeros((3, ik_model.nv))
    last_err = float("inf")

    for it in range(IK_MAX_ITER):
        mujoco.mj_forward(ik_model, ik_data)
        ee = _ee_tip_world(ik_data)
        err_vec = target_world - ee
        err = float(np.linalg.norm(err_vec))
        last_err = err
        if err < EE_TOL:
            if verbose:
                print(f"   IK converged at iter {it}, err={err:.4f}")
            break

        # 그리퍼 끝점에 대한 Jacobian (점의 위치는 ee, body 는 fngr)
        mujoco.mj_jac(ik_model, ik_data, jacp, jacr, ee, EE_BODY_ID)
        J = jacp[:, ARM_DOF_ADR]                    # 3x6

        # Damped least squares :  dq = J^T (J J^T + λ²I)^-1 e
        JJT = J @ J.T + (IK_LAMBDA ** 2) * np.eye(3)
        dq  = J.T @ np.linalg.solve(JJT, err_vec)

        for i, qadr in enumerate(ARM_QPOS_ADR):
            lo, hi = ARM_RANGE[i]
            ik_data.qpos[qadr] = clamp(
                ik_data.qpos[qadr] + IK_STEP * dq[i], lo, hi
            )

    result = [float(ik_data.qpos[qadr]) for qadr in ARM_QPOS_ADR]
    return result, last_err


# ============================================================
# 7. 2D Grid Map + A*
# ============================================================
GRID_RES   = 0.5
GRID_HALF  = 10.0
GRID_N     = int(2 * GRID_HALF / GRID_RES)

OBSTACLES = [
    ( 1.5,  0.5, 0.5, 1.5),
    (-1.5, -2.5, 1.5, 0.5),
    ( 3.5, -2.5, 0.5, 1.5),
    (-3.0,  2.5, 1.0, 0.5),
    ( 0.5, -4.5, 0.5, 0.5),
]
ROBOT_INFLATE = 0.45


def world_to_grid(x, y):
    return int((x + GRID_HALF) / GRID_RES), int((y + GRID_HALF) / GRID_RES)


def grid_to_world(gx, gy):
    return -GRID_HALF + (gx + 0.5) * GRID_RES, -GRID_HALF + (gy + 0.5) * GRID_RES


def build_occupancy() -> np.ndarray:
    occ = np.zeros((GRID_N, GRID_N), dtype=bool)
    for gx in range(GRID_N):
        for gy in range(GRID_N):
            wx, wy = grid_to_world(gx, gy)
            blocked = False
            if (abs(wx) > GRID_HALF - ROBOT_INFLATE - 0.1
                or abs(wy) > GRID_HALF - ROBOT_INFLATE - 0.1):
                blocked = True
            for (cx, cy, sx, sy) in OBSTACLES:
                if (abs(wx - cx) <= sx + ROBOT_INFLATE
                    and abs(wy - cy) <= sy + ROBOT_INFLATE):
                    blocked = True
                    break
            occ[gx, gy] = blocked
    return occ


def nearest_free(occ, cell):
    visited = {cell}
    q = deque([cell])
    while q:
        c = q.popleft()
        if not occ[c]:
            return c
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (c[0] + dx, c[1] + dy)
            if 0 <= nb[0] < GRID_N and 0 <= nb[1] < GRID_N and nb not in visited:
                visited.add(nb)
                q.append(nb)
    return cell


def astar(occ, start, goal):
    if occ[start]:
        start = nearest_free(occ, start)
    if occ[goal]:
        goal = nearest_free(occ, goal)

    def h(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_pq = [(h(start, goal), 0.0, start, None)]
    came = {}
    g_score = {start: 0.0}
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_pq:
        _, g, cur, parent = heapq.heappop(open_pq)
        if cur in came:
            continue
        came[cur] = parent
        if cur == goal:
            path = []
            n = cur
            while n is not None:
                path.append(n)
                n = came[n]
            return list(reversed(path))
        for dx, dy in nbrs:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < GRID_N and 0 <= ny < GRID_N):
                continue
            if occ[nx, ny]:
                continue
            if dx != 0 and dy != 0:
                if occ[cur[0] + dx, cur[1]] or occ[cur[0], cur[1] + dy]:
                    continue
            step = math.hypot(dx, dy)
            tentative = g + step
            nb = (nx, ny)
            if tentative < g_score.get(nb, float("inf")):
                g_score[nb] = tentative
                heapq.heappush(open_pq, (tentative + h(nb, goal), tentative, nb, cur))
    return []


def simplify_path(path):
    if not path:
        return []
    way = []
    for i, (gx, gy) in enumerate(path):
        if i == 0 or i == len(path) - 1:
            way.append(grid_to_world(gx, gy))
            continue
        px, py = path[i - 1]
        nx, ny = path[i + 1]
        if (gx - px, gy - py) != (nx - gx, ny - gy):
            way.append(grid_to_world(gx, gy))
    return way


# ============================================================
# 8. Pure-pursuit
# ============================================================
WP_REACH      = 0.35
GOAL_REACH    = 0.50
KP_YAW        = 1.6
HEADING_SLOW  = 0.6


@dataclass
class PathFollower:
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    wp_idx: int = 0

    def reset(self, wps):
        self.waypoints = list(wps)
        self.wp_idx    = 0

    def is_done(self):
        return self.wp_idx >= len(self.waypoints)

    def step(self, x, y, yaw, goal_xy=None):
        if self.is_done():
            return 0.0, 0.0, True

        while self.wp_idx < len(self.waypoints) - 1:
            wx, wy = self.waypoints[self.wp_idx]
            if math.hypot(wx - x, wy - y) < WP_REACH:
                self.wp_idx += 1
            else:
                break

        wx, wy = self.waypoints[self.wp_idx]
        dx, dy = wx - x, wy - y
        dist   = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        yaw_err    = wrap_angle(target_yaw - yaw)

        if goal_xy is not None:
            gx, gy = goal_xy
            if math.hypot(gx - x, gy - y) < GOAL_REACH:
                return 0.0, 0.0, True

        if abs(yaw_err) > HEADING_SLOW:
            vx   = 0.10
            vyaw = clamp(KP_YAW * yaw_err, -MAX_VYAW, MAX_VYAW)
        else:
            vx   = clamp(MAX_VX * (1 - abs(yaw_err) / HEADING_SLOW * 0.5),
                         0.0, MAX_VX)
            vyaw = clamp(KP_YAW * yaw_err, -MAX_VYAW, MAX_VYAW)

        if self.wp_idx == len(self.waypoints) - 1 and dist < 1.0:
            vx *= max(dist / 1.0, 0.3)

        return vx, vyaw, False


# ============================================================
# 9. FSM 상태 정의
# ============================================================
class S:
    NAV       = "NAV"
    APPROACH  = "APPROACH"
    PRE_GRASP = "PRE_GRASP"   # 큐브 위 0.15 m
    DESCEND   = "DESCEND"     # 큐브 옆 (h=cube_z)
    CLOSE     = "CLOSE"       # 그리퍼 닫기
    LIFT      = "LIFT"        # 0.55 m 까지 들기
    HOLD      = "HOLD"        # 들고 있음
    DONE      = "DONE"


# ============================================================
# 10. APPROACH 파라미터
# ============================================================
APPROACH_DIST = 0.65    # 본체 중심 → 큐브 거리 목표 (m)
APPROACH_TIMEOUT = 6.0  # 초

# Arm IK 타겟 오프셋들 (월드 좌표 기준, 큐브 기준 상대)
PRE_GRASP_DZ  = 0.18   # 큐브 위 18cm
DESCEND_DZ    = 0.02   # 큐브와 거의 같은 높이
LIFT_HEIGHT   = 0.55   # 들어올리는 목표 z

# Arm joint 부드러운 추종을 위한 max delta (rad/sim_step)
ARM_MAX_RATE  = 0.06   # rad / sim step  (sim dt=0.002 가정 → 30 rad/s)


# ============================================================
# 11. 부드러운 팔 추종 (rate-limit)
# ============================================================
_arm_ctrl_state = [0.0] * 6   # 현재 ctrl 값 추적


def init_arm_state_from_ctrl():
    for i, aid in enumerate(ARM_ACT_IDS):
        _arm_ctrl_state[i] = float(data.ctrl[aid])


def step_arm_toward(target_q: List[float], gripper: float, max_rate: float = ARM_MAX_RATE):
    """ctrl 을 max_rate 이하로 target_q 로 한 step 만큼 이동"""
    for i, aid in enumerate(ARM_ACT_IDS):
        cur = _arm_ctrl_state[i]
        d   = target_q[i] - cur
        if   d >  max_rate: d =  max_rate
        elif d < -max_rate: d = -max_rate
        new = cur + d
        lo, hi = ARM_RANGE[i]
        new = clamp(new, lo, hi)
        _arm_ctrl_state[i] = new
        data.ctrl[aid] = new
    data.ctrl[arm_f1x] = gripper


# ============================================================
# 12. Approach goal 계산 - 큐브 정면 d 떨어진 위치
# ============================================================
def compute_approach_goal(start_xy, cube_xy, dist=APPROACH_DIST):
    """시작점에서 큐브 방향으로 |start→cube| - dist 위치"""
    sx, sy = start_xy
    cx, cy = cube_xy
    dx, dy = cx - sx, cy - sy
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return cube_xy
    ux, uy = dx / L, dy / L
    gx = cx - ux * dist
    gy = cy - uy * dist
    return gx, gy


# ============================================================
# 13. 메인
# ============================================================
def main():
    print("=" * 64)
    print("  Spot v2 :  A* + Trot + Arm IK Grasp + FSM")
    print("=" * 64)

    # ───── 1. 경로 계획 ─────
    start_xy = (0.0, 0.0)
    cube_xy  = (float(TARGET_START[0]), float(TARGET_START[1]))
    approach_xy = compute_approach_goal(start_xy, cube_xy, APPROACH_DIST)
    print(f"[Plan] start={start_xy}  cube={cube_xy}  approach_goal={approach_xy}")

    occ = build_occupancy()
    s_cell = world_to_grid(*start_xy)
    g_cell = world_to_grid(*approach_xy)
    cell_path = astar(occ, s_cell, g_cell)
    if not cell_path:
        raise RuntimeError("A* 경로를 찾지 못했습니다.")
    waypoints = simplify_path(cell_path)
    # 정확히 approach_xy 를 마지막 점으로 강제
    waypoints[-1] = approach_xy
    print(f"[A*] cells={len(cell_path)} waypoints={len(waypoints)}")
    for i, w in enumerate(waypoints):
        print(f"   WP[{i:02d}] = ({w[0]:+.2f}, {w[1]:+.2f})")

    follower = PathFollower()
    follower.reset(waypoints)

    # ───── 2. 시뮬레이터 ─────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        reset_target_cube()
        set_arm_pose(ARM_STOW, GRIP_OPEN)
        mujoco.mj_forward(model, data)
        print(f"[Init] cube reset to ({data.qpos[TARGET_QPOS+0]:+.2f}, "
              f"{data.qpos[TARGET_QPOS+1]:+.2f}, {data.qpos[TARGET_QPOS+2]:.2f})")

        # 안정화
        print("[Init] 5000 step 안정화 ...")
        hy0, kn0 = leg_ik(FX0, FZ0)
        for _ in range(5000):
            for cfg in LEGS.values():
                data.ctrl[cfg["hx"]] = 0.0
                data.ctrl[cfg["hy"]] = hy0
                data.ctrl[cfg["kn"]] = kn0
            set_arm_pose(ARM_STOW, GRIP_OPEN)
            mujoco.mj_step(model, data)
        for n in LEGS:
            _prev[n] = dict(hx=0.0, hy=hy0, kn=kn0)
        init_arm_state_from_ctrl()
        viewer.sync()
        print(f"[Init] z={data.qpos[BASE+2]:.3f}  done")

        # ───── 3. 메인 루프 ─────
        dt      = model.opt.timestep
        t       = 0.0
        step_i  = 0
        log_int = int(0.5 / dt)
        RAMP    = 1.5

        state    = S.NAV
        state_t0 = 0.0

        ik_target_world = None   # 현재 IK 목표 (월드 좌표)
        ik_target_q     = [data.ctrl[a] for a in ARM_ACT_IDS]
        ik_gripper      = GRIP_OPEN

        while viewer.is_running():
            # ── 현재 상태 ──
            bx = data.qpos[BASE + 0]
            by = data.qpos[BASE + 1]
            bz = data.qpos[BASE + 2]
            bq = data.qpos[BASE + 3 : BASE + 7]
            _, _, yaw = quat_to_rpy(bq)

            cube_x = data.qpos[TARGET_QPOS + 0]
            cube_y = data.qpos[TARGET_QPOS + 1]
            cube_z = data.qpos[TARGET_QPOS + 2]

            ee_tip = _ee_tip_world(data)

            # ─────────── FSM ───────────
            vx_cmd, vyaw_cmd = 0.0, 0.0

            if state == S.NAV:
                vx_cmd, vyaw_cmd, arrived = follower.step(
                    bx, by, yaw, goal_xy=approach_xy)
                ik_target_q = [ARM_STOW[k] for k in ["sh0","sh1","el0","el1","wr0","wr1"]]
                ik_gripper  = GRIP_OPEN
                if arrived:
                    print(f"[FSM] NAV → APPROACH  pos=({bx:+.2f},{by:+.2f})")
                    state = S.APPROACH
                    state_t0 = t

            elif state == S.APPROACH:
                # 큐브를 본체 정면 APPROACH_DIST 거리에 정밀 정렬
                dx, dy = cube_x - bx, cube_y - by
                yaw_err = wrap_angle(math.atan2(dy, dx) - yaw)
                dist    = math.hypot(dx, dy)
                err     = dist - APPROACH_DIST

                if abs(yaw_err) > 0.12:
                    vx_cmd, vyaw_cmd = 0.0, clamp(1.5 * yaw_err, -1.0, 1.0)
                else:
                    vx_cmd   = clamp(0.4 * err, -0.20, 0.30)
                    vyaw_cmd = clamp(1.2 * yaw_err, -0.6, 0.6)

                ik_gripper = GRIP_OPEN
                ik_target_q = [ARM_SEED[k] for k in ["sh0","sh1","el0","el1","wr0","wr1"]]

                aligned = (abs(err) < 0.08 and abs(yaw_err) < 0.10)
                timeout = (t - state_t0) > APPROACH_TIMEOUT
                if aligned or timeout:
                    why = "aligned" if aligned else "TIMEOUT"
                    print(f"[FSM] APPROACH → PRE_GRASP ({why})  "
                          f"dist={dist:.3f}  yaw_err={math.degrees(yaw_err):+.1f}°")
                    state = S.PRE_GRASP
                    state_t0 = t
                    init_arm_state_from_ctrl()

            elif state == S.PRE_GRASP:
                # 그리퍼 끝을 큐브 위 18 cm 로 (그리퍼 열림)
                tgt = np.array([cube_x, cube_y, cube_z + PRE_GRASP_DZ])
                seed = [ARM_SEED[k] for k in ["sh0","sh1","el0","el1","wr0","wr1"]]
                q, err = solve_arm_ik(tgt, seed=seed)
                ik_target_q = q
                ik_gripper  = GRIP_OPEN
                ik_target_world = tgt

                ee_err = float(np.linalg.norm(_ee_tip_world(data) - tgt))
                if ee_err < 0.05 or (t - state_t0) > 3.0:
                    print(f"[FSM] PRE_GRASP → DESCEND  ee_err={ee_err:.3f}")
                    state = S.DESCEND
                    state_t0 = t

            elif state == S.DESCEND:
                tgt = np.array([cube_x, cube_y, max(cube_z + DESCEND_DZ, 0.04)])
                seed = ik_target_q
                q, err = solve_arm_ik(tgt, seed=seed)
                ik_target_q = q
                ik_gripper  = GRIP_OPEN
                ik_target_world = tgt

                ee_err = float(np.linalg.norm(_ee_tip_world(data) - tgt))
                if ee_err < 0.04 or (t - state_t0) > 3.5:
                    print(f"[FSM] DESCEND → CLOSE  ee_err={ee_err:.3f}")
                    state = S.CLOSE
                    state_t0 = t

            elif state == S.CLOSE:
                # 자세 유지하며 그리퍼만 닫음
                ik_gripper = GRIP_CLOSE
                # ik_target_q 는 이전 단계 그대로
                if (t - state_t0) > 1.2:
                    print(f"[FSM] CLOSE → LIFT  cube_z={cube_z:.3f}")
                    state = S.LIFT
                    state_t0 = t

            elif state == S.LIFT:
                tgt = np.array([cube_x, cube_y, LIFT_HEIGHT])
                seed = ik_target_q
                q, err = solve_arm_ik(tgt, seed=seed)
                ik_target_q = q
                ik_gripper  = GRIP_CLOSE
                ik_target_world = tgt

                if (t - state_t0) > 2.5:
                    if cube_z > 0.20:
                        print(f"[FSM] LIFT → HOLD  ✓ cube_z={cube_z:.3f} (성공)")
                    else:
                        print(f"[FSM] LIFT → HOLD  cube_z={cube_z:.3f} (놓침)")
                    state = S.HOLD
                    state_t0 = t

            elif state == S.HOLD:
                # 큐브 들고 가만히
                ik_gripper = GRIP_CLOSE
                if (t - state_t0) > 3.0:
                    print(f"[FSM] HOLD → DONE  cube_z={cube_z:.3f}")
                    state = S.DONE
                    state_t0 = t

            else:   # DONE
                ik_gripper = GRIP_CLOSE

            # ─────────── 보행 ───────────
            if abs(vx_cmd) < 1e-3 and abs(vyaw_cmd) < 1e-3:
                hold_legs_stand()
            else:
                ramp = min(t / RAMP, 1.0)
                apply_legs(t, vx_cmd * ramp, vyaw_cmd * ramp)

            # ─────────── 팔 ───────────
            step_arm_toward(ik_target_q, ik_gripper)

            mujoco.mj_step(model, data)
            viewer.sync()

            # ─────────── 로그 ───────────
            if step_i % log_int == 0:
                ee = _ee_tip_world(data)
                tgt_str = ""
                if ik_target_world is not None:
                    tgt_str = (f"  ee=({ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:.2f}) "
                               f"→ tgt=({ik_target_world[0]:+.2f},"
                               f"{ik_target_world[1]:+.2f},"
                               f"{ik_target_world[2]:.2f})")
                print(f"[t={t:6.2f}] {state:<10} "
                      f"pos=({bx:+.2f},{by:+.2f},{bz:.2f}) "
                      f"yaw={math.degrees(yaw):+6.1f}° "
                      f"v=({vx_cmd:+.2f},{vyaw_cmd:+.2f})  "
                      f"cube=({cube_x:+.2f},{cube_y:+.2f},{cube_z:.2f})"
                      f"{tgt_str}")

            t      += dt
            step_i += 1
            time.sleep(max(0.0, dt - 0.0001))


if __name__ == "__main__":
    main()
