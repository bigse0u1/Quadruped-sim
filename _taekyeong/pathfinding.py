import heapq, math
import numpy as np

GRID_RES  = 0.5
GRID_MIN  = -25.0
GRID_MAX  =  25.0
GRID_SIZE = int((GRID_MAX - GRID_MIN) / GRID_RES)  # 100

ROBOT_RADIUS = 0.9  # obstacle inflation (m)

# scene_map.xml 건물 목록: (center_x, center_y, half_x, half_y)
_OBSTACLES = [
    (-17.5, 15.5, 4.5, 2.5),
    ( -6.5, 18.5, 2.5, 5.5),
    ( 16.5, 21.0, 5.5, 3.0),
    ( 19.5, 14.5, 2.5, 4.5),
    (-18.5, -3.5, 3.5, 6.5),
    (-13.0, -9.5, 2.0, 1.5),
    (  0.0, -7.0, 5.0, 3.0),
    ( 12.0, -2.0, 4.0, 2.0),
    ( -6.5,-16.5, 2.5, 4.5),
]

# 목적지 (scene_map.xml green_0x 위치)
DESTINATIONS = {
    "1": (-22.0, 23.0),
    "2": ( 14.0, 13.0),
    "3": (-11.0, -2.0),
}

def _w2g(wx, wy):
    gx = int((wx - GRID_MIN) / GRID_RES)
    gy = int((wy - GRID_MIN) / GRID_RES)
    return gx, gy

def _g2w(gx, gy):
    return (GRID_MIN + (gx + 0.5) * GRID_RES,
            GRID_MIN + (gy + 0.5) * GRID_RES)

def _build_grid():
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    pad = ROBOT_RADIUS
    for ox, oy, sx, sy in _OBSTACLES:
        gx0, gy0 = _w2g(ox - sx - pad, oy - sy - pad)
        gx1, gy1 = _w2g(ox + sx + pad, oy + sy + pad)
        gx0 = max(0, gx0); gy0 = max(0, gy0)
        gx1 = min(GRID_SIZE-1, gx1); gy1 = min(GRID_SIZE-1, gy1)
        g[gx0:gx1+1, gy0:gy1+1] = True

    # 횡단보도 통로: 신호등 앞·뒤 각 2m를 포함해 직진만 통과 가능하도록
    # y = -1.0 (신호등 2m 남쪽) ~ 9.5 (도로 북쪽 출구 2m) 구간을 막음.
    # A* 경로가 이 범위에서 x=[-1,1] 통로만 쓰도록 강제 → 비스듬 진입 원천 차단.
    CW_OPEN = 1.0          # 횡단보도 통로 반폭 (m) — 흰색 줄 폭 1.2m에 맞춤
    gy0 = max(0, _w2g(0, -1.0)[1])  # 진입 2m 연장 (신호등 y≈2 기준 2m 남쪽)
    gy1 = min(GRID_SIZE-1, _w2g(0, 9.5)[1])  # 출구 2m 연장 (도로 끝 y=7.5 + 2m)
    gx_l = max(0, _w2g(-CW_OPEN, 0)[0])   # 통로 왼쪽 경계
    gx_r = min(GRID_SIZE, _w2g( CW_OPEN, 0)[0])  # 통로 오른쪽 경계
    g[0:gx_l,         gy0:gy1+1] = True   # 도로 왼쪽 구간
    g[gx_r:GRID_SIZE, gy0:gy1+1] = True   # 도로 오른쪽 구간

    return g

_GRID_BASE = _build_grid()


def astar(start_xy, goal_xy):
    grid = _GRID_BASE.copy()

    sx, sy = _w2g(*start_xy)
    gx, gy = _w2g(*goal_xy)
    sx = max(0, min(GRID_SIZE-1, sx)); sy = max(0, min(GRID_SIZE-1, sy))
    gx = max(0, min(GRID_SIZE-1, gx)); gy = max(0, min(GRID_SIZE-1, gy))

    # 시작/목표 셀이 장애물 안이라도 강제로 열기
    grid[sx, sy] = False
    grid[gx, gy] = False

    def h(x, y): return math.hypot(gx - x, gy - y)

    DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    open_set = [(h(sx, sy), 0.0, sx, sy)]
    came_from = {}
    g_score = {(sx, sy): 0.0}

    while open_set:
        _, g, cx, cy = heapq.heappop(open_set)
        if cx == gx and cy == gy:
            path = []
            node = (cx, cy)
            while node in came_from:
                path.append(_g2w(*node))
                node = came_from[node]
            path.append(_g2w(sx, sy))
            path.reverse()
            return path
        for dx, dy in DIRS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE): continue
            if grid[nx, ny]: continue
            ng = g + math.hypot(dx, dy)
            if ng < g_score.get((nx, ny), float('inf')):
                g_score[(nx, ny)] = ng
                came_from[(nx, ny)] = (cx, cy)
                heapq.heappush(open_set, (ng + h(nx, ny), ng, nx, ny))
    return None


def smooth_path(path, min_dist=3.0):
    if not path:
        return []
    result = [path[0]]
    for pt in path[1:]:
        if math.hypot(pt[0]-result[-1][0], pt[1]-result[-1][1]) >= min_dist:
            result.append(pt)
    if result[-1] != path[-1]:
        result.append(path[-1])
    return result
