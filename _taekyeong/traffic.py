import mujoco

# ── 상수 ──────────────────────────────────────────────────────
TRAFFIC_CYCLE  = 10.0
GREEN_DURATION =  5.0

CW_XS     = [0.0]               # 로봇이 멈추는 횡단보도 (중앙만)
CAR_CW_XS = [-15.0, 0.0, 15.0]  # 차량이 서는 횡단보도 (전체)
CW_X_R  = 1.5
CW_Y_IN = 3.0
CW_Y_OUT= 7.5

# ── 상태 ──────────────────────────────────────────────────────
_model = _data = None
_G_RED_B = _G_GRN_B = _G_RED_T = _G_GRN_T = None
_last_state = None

# 횡단보도 상태 머신: FREE → WAITING → CROSSING
_cw_state = "FREE"
_saw_red   = False   # WAITING 중 빨간불을 본 적 있는가
_prev_tl   = None    # 직전 신호등 상태 (red→green 전환 감지용)

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
    """
    Returns True = 로봇 정지 (횡단보도 대기 중)

    FREE    : 정상 이동
    WAITING : 횡단보도 1m 전 정지, 빨간→초록 전환 대기
    CROSSING: 출발 허가 (횡단보도 끝까지 방해 없음)

    도착 시 이미 초록이면 빨간→초록 사이클을 한 번 기다린 뒤 출발.
    """
    global _cw_state, _saw_red, _prev_tl

    near_cw        = any(abs(robot_x - cx) < CW_X_R for cx in CW_XS)
    at_stop_zone   = near_cw and 1.3 < robot_y < 3.3   # 횡단보도 1m 앞 대기 구역
    past_crosswalk = robot_y > CW_Y_OUT

    # 횡단보도 완전 통과 → 상태 리셋
    if past_crosswalk:
        _cw_state = "FREE"
        _saw_red  = False
        _prev_tl  = tl_state
        return False

    if _cw_state == "FREE":
        if at_stop_zone:
            _cw_state = "WAITING"
            _saw_red  = (tl_state == "red")
            _prev_tl  = tl_state
            print("[횡단보도] 정지선 도달 → 대기 시작")
        else:
            _prev_tl = tl_state
        return _cw_state == "WAITING"   # 방금 WAITING 진입 시 즉시 정지

    elif _cw_state == "WAITING":
        if tl_state == "red":
            _saw_red = True
        red_to_green = (_prev_tl == "red" and tl_state == "green")
        _prev_tl = tl_state
        if red_to_green and _saw_red:
            _cw_state = "CROSSING"
            print("[횡단보도] 빨간→초록 전환 확인 → 출발")
            return False
        return True   # 계속 대기

    else:  # CROSSING
        _prev_tl = tl_state
        return False  # 방해 없음
