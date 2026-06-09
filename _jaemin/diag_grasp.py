"""diag_grasp.py — 그래스핑 격리 진단
로봇을 큐브 앞 standoff 거리에 텔레포트시켜 세운 뒤,
PRE_GRASP→DESCEND IK 를 돌려 그리퍼 끝이 실제로 어디까지 내려가는지 측정한다.
네비게이션을 배제하고 그래스핑 자체만 빠르게 반복 실험.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import mujoco
import main_jaemin as M


def place_robot_in_front_of_cube(standoff=0.60):
    """큐브 정면 standoff 거리에 로봇을 세팅 (yaw 로 큐브 바라봄)."""
    mujoco.mj_resetDataKeyframe(M.model, M.data, 0)
    M.reset_target_cube()
    cube = M.TARGET_START.copy()
    # 큐브 -x 방향에서 접근한다고 가정 (단순화: 큐브 정서쪽)
    bx, by = cube[0] - standoff, cube[1]
    yaw = 0.0  # +x 가 큐브 향함
    M.data.qpos[M.BASE + 0] = bx
    M.data.qpos[M.BASE + 1] = by
    M.data.qpos[M.BASE + 3:M.BASE + 7] = [1, 0, 0, 0]  # yaw=0
    M.set_arm_pose(M.ARM_SEED, M.GRIP_OPEN)
    mujoco.mj_forward(M.model, M.data)


def settle(steps=1500):
    hy0, kn0 = M.leg_ik(M.FX0, M.FZ0)
    for _ in range(steps):
        for cfg in M.LEGS.values():
            M.data.ctrl[cfg["hx"]] = 0.0
            M.data.ctrl[cfg["hy"]] = hy0
            M.data.ctrl[cfg["kn"]] = kn0
        M.set_arm_pose(M.ARM_SEED, M.GRIP_OPEN)
        mujoco.mj_step(M.model, M.data)
    M.init_arm_state_from_ctrl()


def hold_stand(body_drop=0.0):
    """body_drop 만큼 몸체를 낮춘 자세로 다리 고정 (발을 힙에 가깝게)."""
    hy0, kn0 = M.leg_ik(M.FX0, M.FZ0 + body_drop)
    for cfg in M.LEGS.values():
        M.data.ctrl[cfg["hx"]] = 0.0
        M.data.ctrl[cfg["hy"]] = hy0
        M.data.ctrl[cfg["kn"]] = kn0


def run_descend(standoff=0.60, descend_dz=0.02, gripper=None, label="", body_drop=0.0):
    place_robot_in_front_of_cube(standoff)
    settle()
    # squat: 몸체를 body_drop 만큼 낮춰 안정화
    if body_drop > 0:
        for _ in range(800):
            hold_stand(body_drop)
            M.set_arm_pose(M.ARM_SEED, M.GRIP_OPEN)
            mujoco.mj_step(M.model, M.data)
        M.init_arm_state_from_ctrl()

    bx = M.data.qpos[M.BASE + 0]; by = M.data.qpos[M.BASE + 1]; bz = M.data.qpos[M.BASE + 2]
    r, p, y = M.quat_to_rpy(M.data.qpos[M.BASE + 3:M.BASE + 7])
    print(f"\n=== standoff={standoff:.2f}  {label} ===")
    print(f"after settle: base z={bz:.3f}  rpy=({np.degrees(r):+.1f},{np.degrees(p):+.1f},{np.degrees(y):+.1f})")

    cube = M.data.qpos[M.TARGET_QPOS:M.TARGET_QPOS + 3]
    # PRE_GRASP 위치
    tgt_pre = np.array([cube[0], cube[1], cube[2] + 0.18])
    # DESCEND 위치
    tgt_des = np.array([cube[0], cube[1], max(cube[2] + descend_dz, 0.04)])

    seed = [M.ARM_SEED[k] for k in ["sh0", "sh1", "el0", "el1", "wr0", "wr1"]]

    DOWN = np.asarray(APPROACH_AXIS, dtype=float)   # 접근축
    # --- PRE_GRASP ---
    q, _ = M.solve_arm_ik(tgt_pre, seed=seed, approach_axis=DOWN); seed = q
    for _ in range(900):
        hold_stand(body_drop); M.step_arm_toward(q, M.GRIP_OPEN); mujoco.mj_step(M.model, M.data)
    # --- DESCEND ---
    q, _ = M.solve_arm_ik(tgt_des, seed=seed, approach_axis=DOWN); seed = q
    for _ in range(900):
        hold_stand(body_drop); M.step_arm_toward(q, M.GRIP_OPEN); mujoco.mj_step(M.model, M.data)
    ee = M._ee_tip_world(M.data)
    cube_now = M.data.qpos[M.TARGET_QPOS:M.TARGET_QPOS+3].copy()
    print(f"  descend: ee=({ee[0]:.2f},{ee[1]:.2f},{ee[2]:.3f}) "
          f"cube=({cube_now[0]:.2f},{cube_now[1]:.2f},{cube_now[2]:.3f}) "
          f"d_xy={np.hypot(ee[0]-cube_now[0],ee[1]-cube_now[1])*1000:.0f}mm")
    # --- CLOSE (그리퍼만 천천히 닫음) ---
    for _ in range(700):
        hold_stand(body_drop); M.step_arm_toward(q, M.GRIP_CLOSE); mujoco.mj_step(M.model, M.data)
    # --- LIFT (제자리에서 들어올림: 현재 ee x,y 고정, z만 상승) ---
    lift_xy = M._ee_tip_world(M.data)[:2]
    tgt_lift = np.array([lift_xy[0], lift_xy[1], 0.45])
    q, _ = M.solve_arm_ik(tgt_lift, seed=seed, approach_axis=DOWN); seed = q
    for _ in range(1200):
        hold_stand(body_drop); M.step_arm_toward(q, M.GRIP_CLOSE); mujoco.mj_step(M.model, M.data)
    cube_f = M.data.qpos[M.TARGET_QPOS:M.TARGET_QPOS+3].copy()
    ok = cube_f[2] > 0.20
    print(f"  LIFT결과: cube_z={cube_f[2]:.3f}  {'✓ 성공' if ok else '✗ 실패'}  "
          f"cube=({cube_f[0]:.2f},{cube_f[1]:.2f})")


APPROACH_AXIS = (0.0, 0.0, -1.0)   # 기본 top-down

if __name__ == "__main__":
    print("########## 받침대 위 큐브(0.19) top-down 그래스핑 ##########")
    for ee_off in [0.07, 0.10, 0.13]:
        M.EE_LOCAL_OFFSET = np.array([ee_off, 0.0, 0.0])
        for dz in [-0.02, 0.00, 0.02]:
            for so in [0.55, 0.60, 0.65]:
                run_descend(standoff=so, descend_dz=dz, body_drop=0.0,
                            label=f"ee={ee_off} dz={dz:+.2f}")
