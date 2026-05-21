import time
import math
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene_map.xml")
data = mujoco.MjData(model)


def get_act(name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid == -1:
        raise ValueError(f"Actuator not found: {name}")
    return aid


def find_free_joint_qpos_addr():
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[j]
    raise RuntimeError("Free joint not found.")


def quat_to_rpy(q):
    w, x, y, z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(float(np.clip(2*(w*y - z*x), -1, 1)))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


fl_hx, fl_hy, fl_kn = get_act("fl_hx"), get_act("fl_hy"), get_act("fl_kn")
fr_hx, fr_hy, fr_kn = get_act("fr_hx"), get_act("fr_hy"), get_act("fr_kn")
hl_hx, hl_hy, hl_kn = get_act("hl_hx"), get_act("hl_hy"), get_act("hl_kn")
hr_hx, hr_hy, hr_kn = get_act("hr_hx"), get_act("hr_hy"), get_act("hr_kn")

arm_sh0 = get_act("arm_sh0")
arm_sh1 = get_act("arm_sh1")
arm_el0 = get_act("arm_el0")
arm_el1 = get_act("arm_el1")
arm_wr0 = get_act("arm_wr0")
arm_wr1 = get_act("arm_wr1")
arm_f1x = get_act("arm_f1x")

base_qpos_addr = find_free_joint_qpos_addr()

L1x = 0.025
L1z = 0.32
L2  = 0.3365

HY_MIN, HY_MAX = -0.898845, 2.29511
KN_MIN, KN_MAX = -2.7929, -0.2544
HX_MIN, HX_MAX = -0.785398, 0.785398

HY_HOME = 1.04
KN_HOME = -1.8


def fk(hy, kn):
    c1,  s1  = math.cos(hy),    math.sin(hy)
    c12, s12 = math.cos(hy+kn), math.sin(hy+kn)
    fx = c1*L1x - s1*L1z - s12*L2
    fz = -s1*L1x - c1*L1z - c12*L2
    return fx, fz


def ik(fx_t, fz_t, hy0=HY_HOME, kn0=KN_HOME):
    hy, kn = hy0, kn0
    for _ in range(80):
        fx_c, fz_c = fk(hy, kn)
        ex, ez = fx_t - fx_c, fz_t - fz_c
        if abs(ex) < 1e-9 and abs(ez) < 1e-9:
            break
        c1,  s1  = math.cos(hy),    math.sin(hy)
        c12, s12 = math.cos(hy+kn), math.sin(hy+kn)
        J00 = -s1*L1x - c1*L1z - c12*L2
        J01 = -c12*L2
        J10 = -c1*L1x + s1*L1z + s12*L2
        J11 = s12*L2
        det = J00*J11 - J01*J10
        if abs(det) < 1e-10:
            break
        dhy = (J11*ex - J01*ez) / det
        dkn = (-J10*ex + J00*ez) / det
        alpha = 0.5
        hy = clamp(hy + alpha*dhy, HY_MIN, HY_MAX)
        kn = clamp(kn + alpha*dkn, KN_MIN, KN_MAX)
    return hy, kn


FOOT_X0, FOOT_Z0 = fk(HY_HOME, KN_HOME)

# ─── 보행 파라미터 ────────────────────────────────────────
STEP_LEN = 0.20
STEP_H   = 0.10
FREQ     = 1.6
DUTY     = 0.60   # stance 비율 (0~1)

ROLL_KP  = 0.15
PITCH_KP = 0.12
MAX_ROLL_CORR  = 0.06
MAX_PITCH_CORR = 0.04

# 속도 명령 (수정해서 사용)
CMD_VX   = 0.6   # 전진 속도 (양수=앞)
CMD_VYAW = 0.0   # 회전 속도 (양수=좌회전)

LEG_CONFIG = {
    "FL": {"hx": fl_hx, "hy": fl_hy, "kn": fl_kn,
           "phase_offset": 0.0, "side": +1,
           "hip_x": +0.29785, "hip_y": +0.1658},
    "HR": {"hx": hr_hx, "hy": hr_hy, "kn": hr_kn,
           "phase_offset": 0.0, "side": -1,
           "hip_x": -0.29785, "hip_y": -0.1658},
    "FR": {"hx": fr_hx, "hy": fr_hy, "kn": fr_kn,
           "phase_offset": 0.5, "side": -1,
           "hip_x": +0.29785, "hip_y": -0.1658},
    "HL": {"hx": hl_hx, "hy": hl_hy, "kn": hl_kn,
           "phase_offset": 0.5, "side": +1,
           "hip_x": -0.29785, "hip_y": +0.1658},
}


def swing_z(t_swing):
    """swing 중 발 높이: 사인 곡선"""
    return FOOT_Z0 + STEP_H * math.sin(math.pi * t_swing)


def foot_trajectory(phase, vx, vyaw, hip_x, hip_y):
    """
    phase: 0~1
    vx: 전진 속도
    vyaw: 회전 속도 (rad/s)
    hip_x, hip_y: 해당 다리의 hip 위치 (body 기준)
    반환: (dx, fz) — hip 기준 상대 x 변위, 절대 z 높이
    """
    stride     = vx   / FREQ          # 한 주기당 전진 거리
    yaw_stride = vyaw / FREQ          # 한 주기당 회전으로 인한 횡방향 거리

    # 회전 시 바깥쪽 발은 더 크게, 안쪽은 더 작게
    yaw_contrib = hip_y * yaw_stride

    total_stride = stride + yaw_contrib
    half = total_stride / 2.0

    if phase < DUTY:
        # stance: 발이 앞(+half)에서 뒤(-half)로 이동 → 몸이 앞으로
        t_stance = phase / DUTY
        dx = half - total_stride * t_stance
        fz = FOOT_Z0
    else:
        # swing: 발이 뒤(-half)에서 앞(+half)으로 회수
        t_swing = (phase - DUTY) / (1.0 - DUTY)
        dx = -half + total_stride * t_swing
        fz = swing_z(t_swing)

    return dx, fz


def apply_trot(t, roll_corr, pitch_corr, vx, vyaw):
    base_phase = (FREQ * t) % 1.0

    for name, cfg in LEG_CONFIG.items():
        phase = (base_phase + cfg["phase_offset"]) % 1.0

        dx, fz = foot_trajectory(
            phase, vx, vyaw,
            cfg["hip_x"], cfg["hip_y"]
        )

        fx_target = FOOT_X0 + dx

        # pitch 보정: 앞 다리(hip_x>0)는 fz 올리고, 뒷 다리는 내림
        fz_target = fz + cfg["hip_x"] * pitch_corr

        hy, kn = ik(fx_target, fz_target)
        hx_val = clamp(cfg["side"] * roll_corr, HX_MIN, HX_MAX)

        data.ctrl[cfg["hx"]] = hx_val
        data.ctrl[cfg["hy"]] = hy
        data.ctrl[cfg["kn"]] = kn


def set_home_pose():
    mujoco.mj_resetDataKeyframe(model, data, 0)
    print(f"[초기화] FOOT_X0={FOOT_X0:.4f}, FOOT_Z0={FOOT_Z0:.4f}")


def set_arm_stow():
    data.ctrl[arm_sh0] = 0.0
    data.ctrl[arm_sh1] = -3.14
    data.ctrl[arm_el0] = 3.06
    data.ctrl[arm_el1] = 0.0
    data.ctrl[arm_wr0] = 0.0
    data.ctrl[arm_wr1] = 0.0
    data.ctrl[arm_f1x] = -1.57


with mujoco.viewer.launch_passive(model, data) as viewer:
    set_home_pose()
    set_arm_stow()
    mujoco.mj_forward(model, data)

    print("[안정화 중...]")
    for _ in range(3000):
        mujoco.mj_step(model, data)
    viewer.sync()

    bz = data.qpos[base_qpos_addr + 2]
    print(f"[안정화 완료] base z={bz:.4f}")

    RAMP_TIME = 2.0   # 처음 1.5초 동안 서서히 속도 올림

    t          = 0.0
    dt         = model.opt.timestep
    debug_int  = int(2.0 / dt)
    step_count = 0

    while viewer.is_running():
        ramp = min(t / RAMP_TIME, 1.0)
        current_vx   = CMD_VX   * ramp
        current_vyaw = CMD_VYAW * ramp

        quat = data.qpos[base_qpos_addr+3: base_qpos_addr+7]
        roll, pitch, yaw = quat_to_rpy(quat)

        roll_corr  = clamp(-ROLL_KP  * roll,  -MAX_ROLL_CORR,  MAX_ROLL_CORR)
        pitch_corr = clamp(-PITCH_KP * pitch, -MAX_PITCH_CORR, MAX_PITCH_CORR)

        apply_trot(t, roll_corr, pitch_corr, current_vx, current_vyaw)

        mujoco.mj_step(model, data)
        viewer.sync()

        if step_count % debug_int == 0:
            bx = data.qpos[base_qpos_addr]
            by = data.qpos[base_qpos_addr+1]
            bz = data.qpos[base_qpos_addr+2]
            print(f"[t={t:.1f}s] pos=({bx:.3f},{by:.3f},{bz:.3f}) "
                  f"roll={math.degrees(roll):.1f}° "
                  f"pitch={math.degrees(pitch):.1f}° "
                  f"yaw={math.degrees(yaw):.1f}°  "
                  f"vx={current_vx:.2f} vyaw={current_vyaw:.2f}")

        t          += dt
        step_count += 1
        time.sleep(max(0, dt - 0.0001))