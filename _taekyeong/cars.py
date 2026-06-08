import mujoco
from traffic import CAR_CW_XS as CW_XS

# ── 상수 ──────────────────────────────────────────────────────
CAR_SPEED       = 6.0
CAR_SPEED_SLOW  = 2.0
CAR_SPEED_CROSS = 9.0
CAR_KP          = 200.0
CAR_BOUND       = 24.0
CW_SLOW_DIST    = 12.0
CW_STOP_DIST    = 2.5
CW_HALF         = 1.5

# ── 상태 ──────────────────────────────────────────────────────
_data = None
_CARS = []

def init(model, data):
    global _data, _CARS
    _data = data

    def act(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    def jnt(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return model.jnt_qposadr[jid], model.jnt_dofadr[jid]

    c1_qp, c1_qv = jnt("car1_slide")
    c2_qp, c2_qv = jnt("car2_slide")
    c3_qp, c3_qv = jnt("car3_slide")
    _CARS = [
        (act("car1_motor"), c1_qp, c1_qv, -20.0, +1.0),
        (act("car2_motor"), c2_qp, c2_qv, +18.0, -1.0),
        (act("car3_motor"), c3_qp, c3_qv,  -5.0, +1.0),
    ]

def _target_speed(world_x, axis, tl_state):
    for cx in CW_XS:
        dist = (cx - world_x) * axis
        if abs(world_x - cx) < CW_HALF:
            return CAR_SPEED_CROSS
        if 0 < dist <= CW_STOP_DIST:
            return 0.0 if tl_state == "green" else CAR_SPEED_CROSS
        if CW_STOP_DIST < dist <= CW_SLOW_DIST:
            return CAR_SPEED_SLOW
    return CAR_SPEED

def apply(tl_state):
    for act, qp, qv, init_x, axis in _CARS:
        world_x = init_x + axis * _data.qpos[qp]
        if axis > 0 and world_x > CAR_BOUND:
            _data.qpos[qp] = (-CAR_BOUND - init_x) / axis
            _data.qvel[qv] = 0.0
        elif axis < 0 and world_x < -CAR_BOUND:
            _data.qpos[qp] = (CAR_BOUND - init_x) / axis
            _data.qvel[qv] = 0.0
        target = _target_speed(world_x, axis, tl_state)
        if target == 0.0:
            _data.qvel[qv] = 0.0
            _data.ctrl[act] = 0.0
        else:
            _data.ctrl[act] = CAR_KP * (target - _data.qvel[qv])
