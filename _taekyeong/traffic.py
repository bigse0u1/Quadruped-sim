import mujoco

# ── 상수 ──────────────────────────────────────────────────────
TRAFFIC_CYCLE  = 10.0
GREEN_DURATION =  5.0

CW_XS   = [-15.0, 0.0, 15.0]
CW_X_R  = 1.5
CW_Y_IN = 3.0
CW_Y_OUT= 7.5

# ── 상태 ──────────────────────────────────────────────────────
_model = _data = None
_G_RED_B = _G_GRN_B = _G_RED_T = _G_GRN_T = None
_last_state = None
_waiting    = False

def init(model, data):
    global _model, _data, _G_RED_B, _G_GRN_B, _G_RED_T, _G_GRN_T
    _model, _data = model, data
    def gid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    _G_RED_B = gid("red_light_bottom")
    _G_GRN_B = gid("green_light_bottom")
    _G_RED_T = gid("red_light_top")
    _G_GRN_T = gid("green_light_top")

def update(t):
    global _last_state
    state = "green" if (t % TRAFFIC_CYCLE) < GREEN_DURATION else "red"
    if state != _last_state:
        _last_state = state
        on_r  = [1.0, 0.0, 0.0, 1.0];  off_r = [0.3, 0.0, 0.0, 1.0]
        on_g  = [0.0, 1.0, 0.0, 1.0];  off_g = [0.0, 0.3, 0.0, 1.0]
        if state == "green":
            _model.geom_rgba[_G_RED_B] = off_r; _model.geom_rgba[_G_GRN_B] = on_g
            _model.geom_rgba[_G_RED_T] = off_r; _model.geom_rgba[_G_GRN_T] = on_g
        else:
            _model.geom_rgba[_G_RED_B] = on_r;  _model.geom_rgba[_G_GRN_B] = off_g
            _model.geom_rgba[_G_RED_T] = on_r;  _model.geom_rgba[_G_GRN_T] = off_g
        print(f"[신호등] {state.upper()}")
    return state

def check_crosswalk(robot_x, robot_y, tl_state):
    global _waiting
    approaching = any(
        abs(robot_x - cx) < CW_X_R and (CW_Y_IN - 2.5) < robot_y < CW_Y_IN
        for cx in CW_XS
    )
    if approaching and tl_state == "red":
        _waiting = True
    if tl_state == "green" or robot_y > CW_Y_OUT:
        _waiting = False
    return _waiting
