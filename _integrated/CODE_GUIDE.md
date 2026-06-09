# 통합 시나리오 코드 설명서

> Spot 4족보행 로봇이 **자연어 프롬프트**로 지정한 사물을 가져오는 통합 시뮬레이션.
> 출발 → A\* 내비게이션 → **비전 신호등 인지** → 횡단보도 통과 → **사물 파지** → 최종 위치 운반 → 내려놓기.

---

## 1. 전체 개요

### 1.1 시나리오 파이프라인

```
프롬프트("마리오 잡아와")
   └→ 사물 좌표 조회
        └→ A* 경로계획 (출발 → 사물 앞 staging)
             └→ holonomic trot 보행으로 추종
                  └→ [도로] 비전으로 신호등 초록 인지 → 횡단보도 통과
                       └→ 사물 앞 정밀 접근(APPROACH)
                            └→ top-down 팔 IK 하강 → 스냅 + weld 파지
                                 └→ 들어올림(LIFT)
                                      └→ 복귀 A* + 횡단보도 재통과
                                           └→ 최종 위치(0,-19)에서 weld 해제(내려놓기)
```

### 1.2 파일 구성

| 파일 | 역할 |
|------|------|
| `main_task.py` | 통합 컨트롤러 (보행·IK·비전·FSM·CLI 전부) |
| `scene_task.xml` | 씬 정의 = 친구 맵 `scene_map.xml` include + 사물 6개 + 파지 weld |
| `_cam_view.py` | 로봇 카메라 3패널 별창(matplotlib, 별도 프로세스) |
| `assets/<이름>/` | 사물 메쉬(`model.obj`) + 텍스처(`texture.png`) |
| `_taekyeong/` (재사용) | `scene_map.xml`(맵), `pathfinding.py`(A\*), `traffic.py`(신호등), `cars.py`(차량) |

원본 `_taekyeong/`·`_jaemin/` 파일은 **수정하지 않고** 재사용한다.

---

## 2. 사물 정의

[Google Scanned Objects](https://github.com/kevinzakka/mujoco_scanned_objects)의 실물 스캔 메쉬 사용. 초록 플랫폼(윗면 z=0.20) 가장자리에 base가 놓이며, 최대 치수 ~0.13 m로 스케일.

| 키 | 사물 | 구역 | 위치 (x,y) | 도로 횡단 |
|----|------|------|-----------|-----------|
| `mario` | 마리오 | green_01 (북서) | (-22.7, 22.05) | O |
| `yoshi` | 요시 | green_01 | (-21.3, 22.05) | O |
| `android` | 안드로이드 | green_02 (동) | (13.05, 12.3) | O |
| `dino` | 공룡 | green_02 | (13.05, 13.7) | O |
| `elephant` | 코끼리 | green_03 (남) | (-11.4, -3.95) | X |
| `dog` | 강아지 | green_03 | (-10.6, -3.95) | X |

프롬프트는 한/영 모두 인식(`NAME_ALIASES`): 마리오/mario, 요시/yoshi, 안드로이드(로봇)/android, 공룡(다이노)/dino, 코끼리/elephant, 강아지(개)/dog.

---

## 3. `scene_task.xml` 구조

```xml
<mujoco>
  <include file="../_taekyeong/scene_map.xml"/>   <!-- 맵 그대로 -->
  <asset>  <!-- 사물 텍스처/재질/메쉬 (이름 충돌 방지 위해 키별 prefix) -->
    <texture name="mario_tex" file="assets/mario/texture.png"/>          <!-- 모델디렉토리 상대 -->
    <material name="mario_mat" texture="mario_tex"/>
    <mesh name="mario_mesh" file="../../../_integrated/assets/mario/model.obj" scale="..."/>  <!-- meshdir 상대 -->
    ... (6개)
  </asset>
  <worldbody>
    <body name="obj_mario" pos="-22.7 22.05 0.20">
      <freejoint name="mario_free"/>
      <geom type="mesh" mesh="mario_mesh" material="mario_mat" mass="0.2" condim="6"/>
    </body>
    ... (6개)
  </worldbody>
  <equality>  <!-- 파지용 weld, 평소 비활성 -->
    <weld name="grab_mario" body1="arm_link_fngr" body2="obj_mario" active="false" solref="0.02 1"/>
    ... (6개)
  </equality>
</mujoco>
```

**경로 핵심**: 맵을 include하면 `meshdir`이 spot 로봇 assets(`robots/.../assets`)로 잡힌다.
- **텍스처** `file=` 은 모델 디렉토리(`_integrated`) 상대 → `assets/mario/texture.png`
- **메쉬** `file=` 은 `meshdir` 상대 → `../../../_integrated/assets/mario/model.obj`

둘 다 상대경로라 다른 PC에서도 동작한다.

---

## 4. `main_task.py` — 섹션별 설명

### 4.0 모듈 로드 & 런타임 설정
- `scene_task.xml` 로 `model`/`data` 로드. IK 전용 사본 `ik_model`/`ik_data` (실제 시뮬과 분리해 팔 IK 풀이).
- `traffic.GREEN_DURATION=16`, `TRAFFIC_CYCLE=24` — **초록불 연장**(로봇이 도로를 완전히 건널 시간 확보).
- `pathfinding._OBSTACLES += [3개 초록 플랫폼]` 후 그리드 재생성 — A\*가 플랫폼(0.2 m 턱)을 피해 가장자리로 접근.

### 4.1 유틸리티
`get_act / jnt_qadr / jnt_dadr / jnt_range / body_id / eq_id` — 이름→인덱스 조회.
`find_free_qpos`(로봇 free joint), `quat_to_rpy`, `clamp`, `wrap_angle`.

### 4.2 사물 레지스트리
- `Obj` 데이터클래스: `key, body, free(joint), weld(eq), start(위치), stop(staging점)` + 런타임 인덱스.
- `OBJECTS` 6개 정의. `stop`은 사물에서 ~1.6 m 떨어진 **staging point**(플랫폼 인플레이션 밖). 최종 0.62 m 접근은 APPROACH가 담당.
- `reset_objects()`: 키프레임 로드 시 사물 qpos가 0으로 패딩되므로 명시적으로 받침 위에 재배치.
- `parse_prompt(text)`: 프롬프트 문자열에서 사물 키 추출.

### 4.3 다리 기구학 + holonomic 보행
- `fk(hy,kn)` / `leg_ik(fx,fz)`: 한 다리의 2관절(허벅지 hy, 무릎 kn) 정/역기구학(시상면).
- **`foot_traj(phase, vx, vyaw, hip_x, hip_y)`** — 핵심. 정지(stance) 동안 발이 지면에 박힌 채로 몸체에 대해 `-(V + ω×r)` 만큼 쓸려야 미끄럼 없이 추진/회전:
  - `Sx = (vx - vyaw·hip_y)/FREQ` (앞뒤)
  - `Sy = (vyaw·hip_x)/FREQ` (좌우 ← **회전 핵심**)
- `apply_legs(t, vx, vyaw)`: 4다리 trot. 측면 발 오프셋 `dy`를 `hx = asin(dy/L)` 로 변환 → **회전 권한 강화**(데모 초기 6°/s → 20°/s).
- `hold_legs_stand()`: 정지 자세.

> **개선 포인트**: 원래 보행은 회전이 stride 차분에만 의존해 매우 느렸음. holonomic 측면 발배치(hx)를 추가해 회전이 ~3배 빨라짐.

### 4.4 팔 IK + 그래스핑
- `ARM_SEED`(파지 준비 자세) / `ARM_STOW`(접은 자세), `GRIP_OPEN/CLOSE`.
- `step_arm_toward(target_q, gripper, max_rate)`: 팔 관절 + 그리퍼를 **rate-limit**으로 부드럽게 추종(`GRIP_MAX_RATE` 작게 → 물건 안 튕김).
- **`solve_arm_ik(target, seed, approach_axis)`** — Damped Least Squares 팔 IK:
  - `approach_axis` 주면 그리퍼 접근축(local +x)을 그 방향으로 정렬 → **top-down 파지**(`GRASP_AXIS=(0,0,-1)`).
  - 위치(3) + 방향(3) 6-DoF 자코비안으로 풀이. `ik_data`에서 본체 고정·팔만 풀이.
- `_ee_tip_world()`: 그리퍼 끝점 world 좌표(`EE_LOCAL_OFFSET` 적용).

### 4.5 weld 파지/해제
- `attach(obj)`: 그리퍼-사물 상대 자세를 계산해 `model.eq_data`에 기록 후 weld 활성화. **그리퍼가 사물에서 0.16 m 이내일 때만** 부착(멀리서 부착 시 시뮬 폭발 방지).
- `detach(obj)`: weld 비활성(내려놓기).

> **개선 포인트**: 작은 물건은 마찰 파지가 MuJoCo에서 불안정해 자꾸 튕겨나갔음. → 하강 중 사물을 그리퍼 끝으로 **스냅**하고 **soft weld**(`solref="0.02 1"`)로 고정하는 방식으로 전환 → 6개 전부 안정 파지.

### 4.6 비전 — 신호등 인지 (`Vision`)
- `detect_green(robot_x, robot_y)`:
  1. 로봇에서 **가까운 신호등**(남 `TL_BOTTOM` / 북 `TL_TOP`) 선택.
  2. 그 신호등을 바라보는 오프스크린 카메라로 렌더(`mujoco.Renderer`).
  3. **초록 픽셀 수 vs 빨강 픽셀 수** 비교 → 초록 우세하면 green. (운반 사물이 일부 가려도 강건)
- `last_img`: 카메라창(`--cam`)에 띄울 마지막 비전 영상 보관.

> **성능**: 비전 렌더는 무거워서 **0.1초마다(20스텝)만** 호출하고 결과 캐시 → 배속이 횡단보도에서도 유지됨.

### 4.7 횡단보도 게이트 (`CrosswalkGate`)
- **양방향**(남→북, 북→남) 모두 도로 진입 전 정지.
- 상태: `idle → waiting → crossing`. **빨강을 본 뒤 초록(신선한 초록)** 이어야 출발 → 횡단 중 빨강 전환으로 차에 치이는 것 방지.

### 4.8 경로계획 & 추종
- `plan(start, goal)`: 친구 `pathfinding.astar` + `smooth_path`(촘촘히). **도로 횡단 시 횡단보도 중앙선(x=0)을 직진 통과**하도록 경유점 `(0,8.8),(0,1.0)` 강제 삽입.
- `PathFollower`: pure-pursuit. 헤딩 오차 작으면(<0.05 rad) 회전 명령 0 → **직진 흔들림 억제**.

### 4.9 FSM (`main` 루프)
상태: `GO_TO_OBJECT → APPROACH → PRE_GRASP → DESCEND → CLOSE → LIFT → RETURN → PLACE_DOWN → RELEASE → DONE`

| 상태 | 하는 일 |
|------|---------|
| GO_TO_OBJECT | staging까지 A\* 추종, 횡단보도 게이트 적용, 팔 접음 |
| APPROACH | 사물 정면 0.62 m로 정밀 접근(저속 게이트 회피 위해 최소 0.5 m/s) |
| PRE_GRASP | 사물 위 18 cm로 top-down IK |
| DESCEND | 하강하며 사물에 0.16 m 근접 시 **스냅+weld** |
| CLOSE | 그리퍼 닫는 모션(시각용) |
| LIFT | 사물 들어올림 |
| RETURN | (0,-19)로 복귀 A\*, 횡단보도 재통과, 운반 자세 유지 |
| PLACE_DOWN | 팔을 바닥 근처로 하강 |
| RELEASE | weld 해제(내려놓기) → DONE |

`stabilize()`: 시작 시 4000스텝 안정화.

### 4.10 녹화 / 카메라 / 배속 / CLI
- **배속** `--speed=N`: sleep을 `dt/N`로 축소. 고배속(>4)에선 렌더 빈도도 줄여 시뮬이 따라감.
- **카메라창** `--cam`: 별도 프로세스(`_cam_view`)로 body/arm/신호등 3패널 실시간 표시.
- **녹화** `--record`: 추적 카메라 + 비전 합성 프레임을 `frames/`에 PNG 저장.
- **헤드리스** `--headless`: 뷰어 없이 최대 속도 + 결과 출력.

---

## 5. 주요 파라미터 (튜닝 포인트)

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| `FREQ / DUTY / STEP_H` | 2.4 / 0.65 / 0.13 | 보행 주파수·지지율·발 높이 |
| `MAX_VX / MAX_VYAW` | 0.75 / 1.2 | 최대 전진/회전 속도 |
| `APPROACH_DIST` | 0.62 | 파지 접근 정지 거리 |
| `APPROACH_VX_MIN` | 0.50 | 접근 최소 속도(저속 게이트 회피) |
| `GRASP_AXIS` | (0,0,-1) | top-down 접근축 |
| `GRIP_MAX_RATE` | 0.004 | 그리퍼 닫힘 속도(작을수록 부드러움) |
| `traffic.GREEN_DURATION` | 16 | 초록불 길이(횡단 시간 확보) |
| `vision_stride` | 0.1초 | 비전 렌더 주기(배속 유지) |

---

## 6. 실행법

```bash
cd _integrated
python main_task.py                          # 대화형("공룡 잡아와")
python main_task.py --obj=mario              # 직접 지정 (뷰어)
python main_task.py --obj=mario --speed=5    # 5배속
python main_task.py --obj=mario --cam        # 카메라 별창 함께
python main_task.py --obj=mario --speed=5 --cam   # 추천
python main_task.py --headless --obj=mario   # 결과만 빠르게
python main_task.py --record  --obj=mario    # frames/ 에 녹화
```

물건 키: `mario, yoshi, android, dino, elephant, dog`

---

## 7. 개발 중 해결한 핵심 이슈 (요약)

1. **회전이 너무 느림** → holonomic 측면 발배치(hx) 추가.
2. **저속에서 제자리걸음** → APPROACH 최소 속도 0.5 m/s.
3. **작은 물건 파지 실패/폭발** → 스냅 + soft weld, 근접 시에만 부착.
4. **횡단 중 차 충돌** → 초록불 연장 + 양방향 게이트 + 중앙선 직진.
5. **운반 사물이 신호등 가림** → 가까운 신호등 선택 + 초록/빨강 픽셀 비교.
6. **횡단보도에서 배속 풀림** → 비전 렌더를 0.1초 주기로 캐시.
