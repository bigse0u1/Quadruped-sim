# Robot_A_P

MuJoCo 기반 사족보행 모바일 매니퓰레이션 프로젝트

---

## 프로젝트 개요

본 프로젝트는 MuJoCo 시뮬레이션 환경에서  
사족보행 로봇(Quadruped Robot)과 로봇팔(Robot Arm)을 활용한  
자율 이동 및 물체 회수 시스템을 구현하는 것을 목표로 합니다.

로봇은 다음과 같은 작업을 수행합니다.

- 맵 기반 자율 이동
- 최단 경로 탐색
- 사람과 같은 동적 장애물 회피
- 목표 상점 탐색
- 목표 물체 인식 및 회수

본 프로젝트는 다음 분야들을 통합하는 것을 목표로 합니다.

- 사족보행 로봇 제어
- 경로 계획 (Path Planning)
- 로봇 비전 (Robot Vision)
- 모바일 매니퓰레이션 (Mobile Manipulation)
- 자율 로봇 시스템 (Autonomous Robot System)

---

## 주요 기능

### 1. 사족보행 제어

- Trot gait 기반 보행
- Inverse Kinematics (IK) 기반 다리 제어
- Roll / Pitch 안정화
- 속도 명령 기반 gait controller

### 2. Navigation

- 2D 맵 기반 경로 이동
- Waypoint 추종
- A* 기반 최단 경로 탐색 (구현 예정)

### 3. 동적 장애물 처리

- 사람 감지
- 사람 등장 시 정지
- 재이동 / 재경로 탐색

### 4. Mobile Manipulation

- 사족보행 로봇팔 기반 물체 회수
- 목표 물체 grasping

### 5. 시뮬레이션 환경

- 커스텀 MuJoCo 맵
- 건물 및 통로
- 이동하는 사람
- 상점 구역

---

## 프로젝트 시나리오

**예시: "사과 가져와"**

1. **사용자 입력** — `"사과 가져와"`
2. **시스템 판단** — 사과 → 과일가게
3. **경로 생성** — A* 알고리즘 기반 최단 경로 생성
4. **로봇 이동** — 사족보행 기반 목적지 이동
5. **장애물 처리** — 사람이 앞에 있으면 잠시 정지
6. **물체 회수** — 사과 탐지 후 로봇팔로 집기
7. **작업 완료**

---

## 사용 기술

- Python
- MuJoCo
- NumPy
- Quadruped Gait Control
- Inverse Kinematics (IK)
- Path Planning
- Robot Vision (예정)
- Embodied AI

---

## 프로젝트 구조

```
Robot_A_P/
│
├── main.py
├── scene_map.xml
├── robots/
│   └── boston_dynamics_spot/
│
├── assets/
├── maps/
├── planners/
├── controllers/
└── README.md
```

---

## 설치 방법

### 1. Repository Clone

```bash
git clone https://github.com/bigse0u1/Quadruped-sim.git
cd Quadruped-sim
```

### 2. 가상환경 생성

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 라이브러리 설치

```bash
pip install mujoco numpy
```

---

## 실행 방법

```bash
# macOS
mjpython main.py

# Windows / Linux
python main.py
```

---

## 현재 구현 상태

### 구현 완료

- [x] 사족보행 전진 보행
- [x] Trot gait controller
- [x] IK 기반 다리 제어
- [x] Roll / Pitch 안정화
- [x] MuJoCo 환경 구성

### 구현 진행 중

- [ ] A* Path Planning
- [ ] 동적 장애물 회피
- [ ] Vision 기반 물체 탐지
- [ ] 로봇팔 grasping

### 향후 목표

- [ ] 강화학습 기반 locomotion
- [ ] SLAM 연동
- [ ] VLA (Vision-Language-Action) 모델 연동
- [ ] ROS2 연동
- [ ] 실제 로봇 적용

---

## 프로젝트 목표

본 프로젝트는 단순 locomotion 구현보다는 다음과 같은 고수준 자율 시스템 통합에 중점을 둡니다.

- Navigation
- Task Planning
- Dynamic Obstacle Handling
- Mobile Manipulation
- Embodied AI

---

## 라이선스

본 프로젝트는 공개된 MuJoCo 및 로봇 에셋을 사용합니다.  
각 로봇 모델 및 에셋의 라이선스는 원본 Repository를 참고하시기 바랍니다.

---

## 참고 자료

- [MuJoCo](https://mujoco.org)
- [Google DeepMind](https://github.com/google-deepmind)
- [MuJoCo Menagerie (Boston Dynamics Spot Model)](https://github.com/google-deepmind/mujoco_menagerie)
