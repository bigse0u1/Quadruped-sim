# 통합 시나리오 데모 (`_integrated/`)

자연어 프롬프트로 물건을 지정하면, Spot 로봇이 출발 → A* 내비 →
**비전으로 신호등 초록 인지** → 횡단보도 통과(초록 시간 연장) →
해당 물건 파지 → 최종 위치 `(0, -19)` 로 운반 → 내려놓기 까지 수행한다.

## 실행

```bash
cd _integrated
python main_task.py                 # 대화형: "사과 잡아와" 같이 입력
python main_task.py --obj=apple     # 물건 직접 지정 (뷰어, 실시간)
python main_task.py --obj=apple --speed=4   # 4배속 뷰어
python main_task.py --obj=apple --fast      # 5배속 (= --speed=5)
python main_task.py --obj=apple --cam       # 로봇 카메라 별창(body/arm/신호등) 함께 표시
python main_task.py --obj=apple --speed=5 --cam   # 배속 + 카메라창
python main_task.py --headless --obj=apple  # 뷰어 없이 최대 속도 + 결과 출력
python main_task.py --record --obj=apple    # frames/ 에 프레임 저장
```

`--cam`: body_cam / arm_cam / 신호등 비전 3개 패널을 별도 창(matplotlib)으로 실시간 표시.
신호등 비전 패널은 로봇이 횡단보도 근처일 때 갱신됨.

**배속**: `--speed=N` (뷰어 모드). 실시간이 느리면 2~6배속 권장.
고배속에선 렌더 빈도를 자동으로 줄여 시뮬이 따라가게 함. `--headless` 는 sleep 없이 최대 속도.

프롬프트 인식 이름(한/영): 사과/apple, 바나나/banana, 오렌지/orange,
포도/grape, 레몬/lemon, 토마토/tomato.

## 물건 배치 (초록 구역당 2개)

| 물건 | 구역 | 위치 | 도로 횡단 |
|------|------|------|-----------|
| 사과·바나나 | green_01 (북서) | (-22.7,22.05), (-21.3,22.05) | O |
| 오렌지·포도 | green_02 (동)   | (13.05,12.3), (13.05,13.7)  | O |
| 레몬·토마토 | green_03 (남)   | (-11.4,-3.9), (-10.6,-3.9)  | X |

물건은 초록 플랫폼(윗면 z=0.20) 가장자리에 z=0.24 로 놓여 팔이 닿는 높이.

## 구성 요소

- **보행**: holonomic foot-placement trot (jaemin) — 측면 발배치로 회전 강화
- **내비**: 친구(taekyeong) `pathfinding` A* 재사용 + 초록 플랫폼을 장애물로 추가.
  도로 횡단 시 횡단보도 중앙선(x=0)을 직진 통과하도록 경유점 강제 삽입.
- **비전 신호등**: 로봇 시점 오프스크린 렌더 → 가까운 신호등(남/북)을 보고
  초록 픽셀 vs 빨강 픽셀 비교로 ON 판정 (운반 물건 가림에도 강건).
- **횡단보도 게이트**: 양방향. 도로 진입 전 정지, 빨강→초록(신선한 초록) 인지 시 출발.
  초록 시간 16초로 연장(`traffic.GREEN_DURATION`).
- **파지**: top-down 6-DoF IK 접근 후 그리퍼 끝으로 물건 스냅 + `weld` equality 부착
  (작은 물건 안정 파지). 내려놓을 때 weld 해제.

## 검증 (headless)

6개 물건 전부 end-to-end 성공: 프롬프트→내비→(횡단 비전)→파지→운반→내려놓기.

## 메모

- **맵은 친구의 `_taekyeong/scene_map.xml` 을 그대로 `<include>`** 하여 사용 (복제 아님).
  `scene_task.xml` 은 그 위에 물건 6개 + 파지 weld 만 추가 → 맵이 바뀌면 자동 반영.
- 원본 `_taekyeong/`, `_jaemin/` 파일은 수정하지 않음.
- `traffic`/`pathfinding` 모듈의 상수(초록 시간, 장애물 목록)는 import 후 런타임에 조정.
