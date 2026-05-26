import time, math, enum
import numpy as np
import mujoco, mujoco.viewer

# ──────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("scene_map.xml")
data  = mujoco.MjData(model)

def get_act(name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid == -1: raise ValueError(f"Actuator not found: {name}")
    return aid

def find_free_qpos():
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[j]
    raise RuntimeError("Free joint not found")

def quat_to_rpy(q):
    w,x,y,z = q
    roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(float(np.clip(2*(w*y-z*x),-1,1)))
    yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return roll, pitch, yaw

def clamp(v, lo, hi): return float(np.clip(v, lo, hi))

# ──────────────────────────────────────────────────────────────
#  액추에이터
# ──────────────────────────────────────────────────────────────
fl_hx,fl_hy,fl_kn = get_act("fl_hx"),get_act("fl_hy"),get_act("fl_kn")
fr_hx,fr_hy,fr_kn = get_act("fr_hx"),get_act("fr_hy"),get_act("fr_kn")
hl_hx,hl_hy,hl_kn = get_act("hl_hx"),get_act("hl_hy"),get_act("hl_kn")
hr_hx,hr_hy,hr_kn = get_act("hr_hx"),get_act("hr_hy"),get_act("hr_kn")
arm_sh0=get_act("arm_sh0"); arm_sh1=get_act("arm_sh1")
arm_el0=get_act("arm_el0"); arm_el1=get_act("arm_el1")
arm_wr0=get_act("arm_wr0"); arm_wr1=get_act("arm_wr1")
arm_f1x=get_act("arm_f1x")
BASE = find_free_qpos()

# ──────────────────────────────────────────────────────────────
#  기구학
# ──────────────────────────────────────────────────────────────
L1x=0.025; L1z=0.32; L2=0.3365
HY_MIN,HY_MAX = -0.898845, 2.29511
KN_MIN,KN_MAX = -2.7929,  -0.2544
HX_MIN,HX_MAX = -0.785398, 0.785398
HY_HOME=1.04; KN_HOME=-1.8

def fk(hy, kn):
    c1,s1   = math.cos(hy),  math.sin(hy)
    c12,s12 = math.cos(hy+kn), math.sin(hy+kn)
    return c1*L1x-s1*L1z-s12*L2, -s1*L1x-c1*L1z-c12*L2

def ik(fx_t, fz_t, hy=HY_HOME, kn=KN_HOME):
    for _ in range(100):
        fx,fz = fk(hy,kn)
        ex,ez = fx_t-fx, fz_t-fz
        if abs(ex)<1e-9 and abs(ez)<1e-9: break
        c1,s1   = math.cos(hy),  math.sin(hy)
        c12,s12 = math.cos(hy+kn), math.sin(hy+kn)
        J00=-s1*L1x-c1*L1z-c12*L2; J01=-c12*L2
        J10=-c1*L1x+s1*L1z+s12*L2; J11= s12*L2
        det=J00*J11-J01*J10
        if abs(det)<1e-10: break
        hy=clamp(hy+0.5*(J11*ex-J01*ez)/det, HY_MIN,HY_MAX)
        kn=clamp(kn+0.5*(-J10*ex+J00*ez)/det, KN_MIN,KN_MAX)
    return hy,kn

FX0, FZ0 = fk(HY_HOME, KN_HOME)
print(f"[IK 기준] FX0={FX0:.4f}  FZ0={FZ0:.4f}")

# ──────────────────────────────────────────────────────────────
#  보행 파라미터  ← 여기만 수정해서 튜닝
# ──────────────────────────────────────────────────────────────
FREQ      = 2.0    # 보행 주파수 (Hz)
DUTY      = 0.70   # stance 비율
STEP_H    = 0.2   # 발 리프트 높이 (m)

CMD_VX    = 0.8    # 전진 속도 (m/s) — 양수=앞
CMD_VYAW  = 0.0    # 회전 (rad/s)    — 양수=좌

SWING_FILTER = 0.55   # swing 필터 (stance는 필터 없음)

# ──────────────────────────────────────────────────────────────
#  Trot 대각선 페어
#   Pair A (offset 0.0): FL(앞왼) + HR(뒤오른)
#   Pair B (offset 0.5): FR(앞오른) + HL(뒤왼)
# ──────────────────────────────────────────────────────────────
LEGS = {
    "FL": dict(hx=fl_hx, hy=fl_hy, kn=fl_kn, off=0.0, side=+1, hx_pos=+0.298, hy_pos=+0.166),
    "HR": dict(hx=hr_hx, hy=hr_hy, kn=hr_kn, off=0.0, side=-1, hx_pos=-0.298, hy_pos=-0.166),
    "FR": dict(hx=fr_hx, hy=fr_hy, kn=fr_kn, off=0.5, side=-1, hx_pos=+0.298, hy_pos=-0.166),
    "HL": dict(hx=hl_hx, hy=hl_hy, kn=hl_kn, off=0.5, side=+1, hx_pos=-0.298, hy_pos=+0.166),
}

# 필터 상태
_prev = {n: dict(hx=0.0, hy=HY_HOME, kn=KN_HOME) for n in LEGS}

# ──────────────────────────────────────────────────────────────
#  발 궤적
# ──────────────────────────────────────────────────────────────
def foot_traj(phase, vx, vyaw, hip_y):
    """
    stance: 앞(+half) → 뒤(-half) 선형 sweep  → 몸 앞으로
    swing : 뒤(-half) → 앞(+half) sin 리프트
    """
    stride = vx / FREQ
    yaw_d  = vyaw / FREQ * hip_y     # 회전 시 좌우 기여
    S      = stride + yaw_d           # 한 주기 총 이동량
    half   = S / 2.0

    if phase < DUTY:                  # ── stance ──
        t  = phase / DUTY
        dx = half - S * t             # +half → -half
        dz = 0.0
    else:                             # ── swing ──
        t  = (phase - DUTY) / (1.0 - DUTY)
        dx = -half + S * t
        dz = STEP_H * math.sin(math.pi * t)

    return FX0 + dx, FZ0 + dz

# ──────────────────────────────────────────────────────────────
#  컨트롤 적용
# ──────────────────────────────────────────────────────────────
def set_arm_stow():
    data.ctrl[arm_sh0]=0.0; data.ctrl[arm_sh1]=-3.14
    data.ctrl[arm_el0]=3.06; data.ctrl[arm_el1]=0.0
    data.ctrl[arm_wr0]=0.0; data.ctrl[arm_wr1]=0.0
    data.ctrl[arm_f1x]=-1.57

def apply_legs(t_sim, vx, vyaw):
    for name, cfg in LEGS.items():
        phase = (FREQ * t_sim + cfg["off"]) % 1.0
        in_stance = phase < DUTY

        fx_t, fz_t = foot_traj(phase, vx, vyaw, cfg["hy_pos"])
        hy_t, kn_t = ik(fx_t, fz_t)
        hx_t = 0.0   # roll 보정 제거 — 단순화

        if in_stance:
            # stance: 필터 없이 직접 커맨드 → 추진력 최대
            data.ctrl[cfg["hx"]] = hx_t
            data.ctrl[cfg["hy"]] = hy_t
            data.ctrl[cfg["kn"]] = kn_t
            _prev[name] = dict(hx=hx_t, hy=hy_t, kn=kn_t)
        else:
            # swing: 약한 필터로 부드럽게
            p = _prev[name]
            hx = SWING_FILTER*p["hx"] + (1-SWING_FILTER)*hx_t
            hy = SWING_FILTER*p["hy"] + (1-SWING_FILTER)*hy_t
            kn = SWING_FILTER*p["kn"] + (1-SWING_FILTER)*kn_t
            data.ctrl[cfg["hx"]] = hx
            data.ctrl[cfg["hy"]] = hy
            data.ctrl[cfg["kn"]] = kn
            _prev[name] = dict(hx=hx, hy=hy, kn=kn)

# ──────────────────────────────────────────────────────────────
#  메인
# ──────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    set_arm_stow()
    mujoco.mj_forward(model, data)

    # 안정화: 홈포즈 커맨드 유지한 채 5000스텝
    print("[안정화 중...]")
    hy0, kn0 = ik(FX0, FZ0)
    for _ in range(5000):
        for cfg in LEGS.values():
            data.ctrl[cfg["hx"]] = 0.0
            data.ctrl[cfg["hy"]] = hy0
            data.ctrl[cfg["kn"]] = kn0
        set_arm_stow()
        mujoco.mj_step(model, data)

    # 필터 초기값 세팅
    for n in LEGS: _prev[n] = dict(hx=0.0, hy=hy0, kn=kn0)

    bz = data.qpos[BASE+2]
    print(f"[안정화 완료] z={bz:.4f}")
    viewer.sync()

    dt      = model.opt.timestep
    t       = 0.0
    step_i  = 0
    log_int = int(2.0 / dt)

    # 속도 ramp
    RAMP = 1.5   # 초

    while viewer.is_running():
        ramp = min(t / RAMP, 1.0)
        vx   = CMD_VX   * ramp
        vyaw = CMD_VYAW * ramp

        apply_legs(t, vx, vyaw)
        set_arm_stow()
        mujoco.mj_step(model, data)
        viewer.sync()

        if step_i % log_int == 0:
            bx = data.qpos[BASE]
            by = data.qpos[BASE+1]
            bz = data.qpos[BASE+2]
            q  = data.qpos[BASE+3:BASE+7]
            roll, pitch, yaw = quat_to_rpy(q)
            phase_fl = (FREQ*t + LEGS["FL"]["off"]) % 1.0
            phase_fr = (FREQ*t + LEGS["FR"]["off"]) % 1.0
            print(f"[t={t:5.1f}s] "
                  f"pos=({bx:+.3f},{by:+.3f},{bz:.3f}) "
                  f"roll={math.degrees(roll):+5.1f}° "
                  f"pitch={math.degrees(pitch):+5.1f}°  "
                  f"vx={vx:.2f}  "
                  f"FL:{phase_fl:.2f}{'S' if phase_fl<DUTY else 'W'} "
                  f"FR:{phase_fr:.2f}{'S' if phase_fr<DUTY else 'W'}")

        t      += dt
        step_i += 1
        time.sleep(max(0, dt - 0.0001))
