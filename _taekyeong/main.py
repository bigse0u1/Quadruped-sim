import time, math
import numpy as np
import mujoco, mujoco.viewer
from multiprocessing import Process, Queue as MPQueue
import _cam_worker
import traffic
import cars
import pathfinding

# ── 보행 상수 ─────────────────────────────────────────────────
L1x=0.025; L1z=0.32; L2=0.3365
HY_MIN,HY_MAX = -0.898845, 2.29511
KN_MIN,KN_MAX = -2.7929,  -0.2544
HY_HOME=1.04; KN_HOME=-1.8

FREQ         = 2.8
DUTY         = 0.70
STEP_H       = 0.22
CMD_VX       = 1.5
CMD_VYAW     = 0.0
SWING_FILTER = 0.55
HX_GAIN_YAW  = 0.05

# ── 유틸 함수 ─────────────────────────────────────────────────
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

# ── 기구학 ────────────────────────────────────────────────────
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

# ── 보행 제어 ─────────────────────────────────────────────────
def foot_traj(phase, vx, vyaw, hip_y):
    stride = vx / FREQ
    S      = stride - vyaw / FREQ * hip_y   # jaemin 기준: vyaw>0=좌회전
    half   = S / 2.0
    if phase < DUTY:
        t  = phase / DUTY
        return FX0 + half - S * t, FZ0
    else:
        t  = (phase - DUTY) / (1.0 - DUTY)
        return FX0 - half + S * t, FZ0 + STEP_H * math.sin(math.pi * t)

def set_arm_stow():
    data.ctrl[arm_sh0]=0.0; data.ctrl[arm_sh1]=-3.14
    data.ctrl[arm_el0]=3.06; data.ctrl[arm_el1]=0.0
    data.ctrl[arm_wr0]=0.0; data.ctrl[arm_wr1]=0.0
    data.ctrl[arm_f1x]=-1.57

def apply_legs(t_sim, vx, vyaw):
    for name, cfg in LEGS.items():
        phase = (FREQ * t_sim + cfg["off"]) % 1.0
        fx_t, fz_t = foot_traj(phase, vx, vyaw, cfg["hy_pos"])
        hy_t, kn_t = ik(fx_t, fz_t)
        hx_t = clamp(HX_GAIN_YAW * vyaw * cfg["side"], -0.785398, 0.785398)
        if phase < DUTY:
            data.ctrl[cfg["hx"]] = hx_t
            data.ctrl[cfg["hy"]] = hy_t
            data.ctrl[cfg["kn"]] = kn_t
            _prev[name] = dict(hx=hx_t, hy=hy_t, kn=kn_t)
        else:
            p = _prev[name]
            hx = SWING_FILTER*p["hx"] + (1-SWING_FILTER)*hx_t
            hy = SWING_FILTER*p["hy"] + (1-SWING_FILTER)*hy_t
            kn = SWING_FILTER*p["kn"] + (1-SWING_FILTER)*kn_t
            data.ctrl[cfg["hx"]] = hx
            data.ctrl[cfg["hy"]] = hy
            data.ctrl[cfg["kn"]] = kn
            _prev[name] = dict(hx=hx, hy=hy, kn=kn)

# ── 메인 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    model = mujoco.MjModel.from_xml_path("scene_map.xml")
    data  = mujoco.MjData(model)

    traffic.init(model, data)
    cars.init(model, data)

    # 목적지 선택
    dest_input = input("목적지를 선택하세요 (1/2/3): ").strip()
    if dest_input not in pathfinding.DESTINATIONS:
        print(f"잘못된 입력 '{dest_input}' → 1번으로 설정")
        dest_input = "1"
    GOAL = pathfinding.DESTINATIONS[dest_input]
    print(f"[목표] {dest_input}번 목적지 → {GOAL}")

    fl_hx,fl_hy,fl_kn = get_act("fl_hx"),get_act("fl_hy"),get_act("fl_kn")
    fr_hx,fr_hy,fr_kn = get_act("fr_hx"),get_act("fr_hy"),get_act("fr_kn")
    hl_hx,hl_hy,hl_kn = get_act("hl_hx"),get_act("hl_hy"),get_act("hl_kn")
    hr_hx,hr_hy,hr_kn = get_act("hr_hx"),get_act("hr_hy"),get_act("hr_kn")
    arm_sh0=get_act("arm_sh0"); arm_sh1=get_act("arm_sh1")
    arm_el0=get_act("arm_el0"); arm_el1=get_act("arm_el1")
    arm_wr0=get_act("arm_wr0"); arm_wr1=get_act("arm_wr1")
    arm_f1x=get_act("arm_f1x")
    BASE = find_free_qpos()

    LEGS = {
        "FL": dict(hx=fl_hx, hy=fl_hy, kn=fl_kn, off=0.0, side=+1, hx_pos=+0.298, hy_pos=+0.166),
        "HR": dict(hx=hr_hx, hy=hr_hy, kn=hr_kn, off=0.0, side=-1, hx_pos=-0.298, hy_pos=-0.166),
        "FR": dict(hx=fr_hx, hy=fr_hy, kn=fr_kn, off=0.5, side=-1, hx_pos=+0.298, hy_pos=-0.166),
        "HL": dict(hx=hl_hx, hy=hl_hy, kn=hl_kn, off=0.5, side=+1, hx_pos=-0.298, hy_pos=+0.166),
    }
    _prev = {n: dict(hx=0.0, hy=HY_HOME, kn=KN_HOME) for n in LEGS}

    print(f"[IK 기준] FX0={FX0:.4f}  FZ0={FZ0:.4f}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        set_arm_stow()
        mujoco.mj_forward(model, data)

        print("[안정화 중...]")
        hy0, kn0 = ik(FX0, FZ0)
        for _ in range(5000):
            for cfg in LEGS.values():
                data.ctrl[cfg["hx"]] = 0.0
                data.ctrl[cfg["hy"]] = hy0
                data.ctrl[cfg["kn"]] = kn0
            set_arm_stow()
            mujoco.mj_step(model, data)

        for n in LEGS: _prev[n] = dict(hx=0.0, hy=hy0, kn=kn0)
        print(f"[안정화 완료] z={data.qpos[BASE+2]:.4f}")
        viewer.sync()

        # A* 경로 계산
        start_xy = (data.qpos[BASE], data.qpos[BASE+1])
        raw_path  = pathfinding.astar(start_xy, GOAL)
        waypoints = pathfinding.smooth_path(raw_path) if raw_path else []

        # 도로를 건너는 경로라면 직진 진입 웨이포인트 삽입.
        # pathfinding 장애물이 y=1.0부터 통로를 막지만, smooth_path가
        # 포인트를 합칠 수 있으므로 (0,0.8)·(0,2.3) 두 점을 명시적으로 고정.
        if any(wy > 3.0 for _, wy in waypoints):
            approach_wps = [(0.0, 0.8), (0.0, 2.3)]
            new_wps, injected = [], False
            for wp in waypoints:
                if not injected and wp[1] > 0.6:
                    new_wps.extend(approach_wps)
                    injected = True
                    if wp[1] > 2.8:          # 도로 위 포인트는 그대로 유지
                        new_wps.append(wp)
                    # 0.6 < y <= 2.8 인 기존 포인트는 approach_wps 로 대체
                else:
                    new_wps.append(wp)
            waypoints = new_wps
            print("[경로] 횡단보도 직진 진입 웨이포인트 삽입 완료")

        wp_idx    = 0
        arrived   = not bool(waypoints)
        prev_wp   = start_xy   # Stanley용: 직전 웨이포인트
        if waypoints:
            print(f"[경로] 웨이포인트 {len(waypoints)}개 생성 완료")
            # 도로(y=2.5~7.5) 부근 웨이포인트 확인
            road_wps = [(round(x,1), round(y,1)) for x,y in waypoints if 0.5 < y < 9.0]
            print(f"  └ 도로 부근 웨이포인트: {road_wps}")
        else:
            print("[경로] 경로를 찾지 못했습니다")

        # A* 경로를 바닥에 빨간 선으로 표시
        if raw_path and len(raw_path) > 1:
            for i in range(len(raw_path) - 1):
                if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                    break
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    0.04,
                    [raw_path[i][0],   raw_path[i][1],   0.05],
                    [raw_path[i+1][0], raw_path[i+1][1], 0.05],
                )
                viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba[:] = [1, 0, 0, 1]
                viewer.user_scn.ngeom += 1
            viewer.sync()
            print(f"[시각화] 경로 선분 {viewer.user_scn.ngeom}개")

        _track_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body"))
        print(f"[카메라] tracking body id={_track_id} / total={model.nbody}")
        if 0 <= _track_id < model.nbody:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = _track_id
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -20.0

        renderer     = mujoco.Renderer(model, height=360, width=480)
        CAM_INTERVAL = 10
        cam_queue    = MPQueue(maxsize=1)
        cam_proc     = Process(target=_cam_worker.run, args=(cam_queue,), daemon=True)
        cam_proc.start()

        dt      = model.opt.timestep
        t       = 0.0
        step_i  = 0
        log_int = int(2.0 / dt)
        RAMP    = 1.5

        while viewer.is_running():
            ramp = min(t / RAMP, 1.0)
            bx = data.qpos[BASE]
            by = data.qpos[BASE+1]

            tl_state = traffic.update(t)

            # A* 웨이포인트 추종 (Stanley 컨트롤러)
            if arrived or not waypoints:
                vx, vyaw = 0.0, 0.0
            else:
                wp_x, wp_y = waypoints[wp_idx]
                dx = wp_x - bx
                dy = wp_y - by
                dist = math.hypot(dx, dy)
                _, _, cur_yaw = quat_to_rpy(data.qpos[BASE+3:BASE+7])

                is_last   = (wp_idx == len(waypoints) - 1)
                threshold = 1.5 if is_last else 1.0
                if dist < threshold:
                    if is_last:
                        arrived = True
                        print("[도착!]")
                        vx, vyaw = 0.0, 0.0
                    else:
                        prev_wp = (wp_x, wp_y)   # 세그먼트 시작점 갱신
                        wp_idx += 1
                        vx, vyaw = 0.0, 0.0
                else:
                    # ── 경로 세그먼트 기반 Stanley ──
                    px, py   = prev_wp
                    seg_head = math.atan2(wp_y - py, wp_x - px)

                    # 횡방향 오차: 양수 = 로봇이 경로 오른쪽 → 왼쪽(+vyaw) 보정
                    cross = (bx - px)*math.sin(seg_head) - (by - py)*math.cos(seg_head)

                    # 헤딩 오차
                    head_err = seg_head - cur_yaw
                    while head_err >  math.pi: head_err -= 2*math.pi
                    while head_err < -math.pi: head_err += 2*math.pi

                    # Stanley 합산 (헤딩 + 횡방향 보정)
                    ct_corr   = float(np.clip(0.6 * cross, -0.5, 0.5))
                    total_err = float(np.clip(head_err + ct_corr, -math.pi, math.pi))

                    if abs(total_err) > 0.4:
                        vx   = CMD_VX * ramp * 0.3
                        vyaw = float(np.clip(1.5 * total_err, -1.0, 1.0))
                    else:
                        vx   = CMD_VX * ramp
                        vyaw = float(np.clip(1.2 * total_err, -0.8, 0.8))

            # 횡단보도 대기 (빨간→초록 전환 확인 전까지 완전 정지)
            if traffic.check_crosswalk(bx, by, tl_state):
                vx   = 0.0
                vyaw = 0.0

            apply_legs(t, vx, vyaw)
            set_arm_stow()
            cars.apply(tl_state)
            mujoco.mj_step(model, data)
            viewer.sync()

            if step_i % CAM_INTERVAL == 0 and not cam_queue.full():
                renderer.update_scene(data, camera="body_cam")
                body_img = renderer.render().copy()
                renderer.update_scene(data, camera="arm_cam")
                arm_img  = renderer.render().copy()
                cam_queue.put_nowait((body_img, arm_img))

            if step_i % log_int == 0:
                bz = data.qpos[BASE+2]
                roll, pitch, _ = quat_to_rpy(data.qpos[BASE+3:BASE+7])
                print(f"[t={t:5.1f}s] pos=({bx:+.2f},{by:+.2f},{bz:.2f}) "
                      f"tl={tl_state}  vx={vx:.2f}")

            t      += dt
            step_i += 1
            time.sleep(max(0, dt - 0.0001))
