import time, math
import numpy as np
import mujoco, mujoco.viewer
from multiprocessing import Process, Queue as MPQueue
import _cam_worker
import traffic
import cars

# ── 보행 상수 ─────────────────────────────────────────────────
L1x=0.025; L1z=0.32; L2=0.3365
HY_MIN,HY_MAX = -0.898845, 2.29511
KN_MIN,KN_MAX = -2.7929,  -0.2544
HY_HOME=1.04; KN_HOME=-1.8

FREQ         = 2.0
DUTY         = 0.70
STEP_H       = 0.2
CMD_VX       = 0.8
CMD_VYAW     = 0.0
SWING_FILTER = 0.55

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
    S      = stride + vyaw / FREQ * hip_y
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
        if phase < DUTY:
            data.ctrl[cfg["hx"]] = 0.0
            data.ctrl[cfg["hy"]] = hy_t
            data.ctrl[cfg["kn"]] = kn_t
            _prev[name] = dict(hx=0.0, hy=hy_t, kn=kn_t)
        else:
            p = _prev[name]
            hx = SWING_FILTER*p["hx"] + (1-SWING_FILTER)*0.0
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

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("body").id
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
            vx   = CMD_VX * ramp
            vyaw = CMD_VYAW * ramp

            tl_state = traffic.update(t)

            bx = data.qpos[BASE]
            by = data.qpos[BASE+1]
            if traffic.check_crosswalk(bx, by, tl_state):
                vx = 0.0

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
