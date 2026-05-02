#### Mujoco, Pinocchio 시뮬레이션 프레임워크 (`mj_sim`)

이 레포지토리는 MuJoCo와 Pinocchio를 활용한 로봇 제어 프레임워크입니다. `mj_sim` 폴더 내에 로봇의 종류에 따라 모듈화되어 있습니다.

---

### 1. Manipulator (`mj_sim/manipulator/`)

#### Refactored Scripts
기존 스터디 파일들을 `core`, `control`, `utils` 패키지로 모듈화하여 재작성한 버전입니다.
- `ur5e_dls_remake.ipynb`: DLS 제어 기반 Remake 버전
- `ur5e_impedance_remake.ipynb`: Impedance 제어 기반 Remake 버전

#### Jupyter Study Files (`jupyter_study_files/`)
초기 학습 및 기능 테스트를 위한 스크립트들입니다.
- **순수 무조코**: `ur5e_mj_dls.ipynb`
- **순수 DLS + pinocchio**: `ur5e_mj2pin_pure_dls.ipynb`
- **PTP 궤적 quintic, cubic**: `ur5e_mj2pin_PTP.ipynb`
- **waypoints 궤적 cubic**: `ur5e_mj2pin_Spline.ipynb`
- **cubic + 6D control**: `ur5e_mj2pin_Spline_6D.ipynb`
- **IK를 QP로 변경**: `ur5e_mj2pin_ik_qp.ipynb`
- **DLS + RRT***: `ur5e_mj2pin_Spline_6D_RRT.ipynb`
- **OCS impedance control**: `ur5e_mj2pin_impedance.ipynb`

---

### 2. Mobile (`mj_sim/mobile/`)

- **LQR error dynamics**: `husky_LQR_tracking.ipynb`
- **MPC**: 

---

### 3. Humanoid (`mj_sim/humanoid/`)

- **Main Simulation**: `g1_main.ipynb` (G1 로봇)
---

### 4. Quadruped (`mj_sim/quarduped/`) -> Copy 

- **Main Simulation**: `go2_main.ipynb` (Go2 로봇)
- **Test Scripts**: `go2_test.py`, `test.py`

---

### How to Make New 
새로운 로봇이나 시뮬레이션 환경을 구성할 때의 가이드입니다.
1. `core`, `utils`를 복사한다.
2. fixed base, floating base인지에 따라서 컨테이너 클래스를 선택 후 내부 슬라이싱 수정 
3. pinocchio wrapper의 FRAMES는 manipulation 대상이므로 알맞은 URDF LINK NAME으로 수정
4. init 부분의 SE3, SO3 부분 수정 
5. mujoco kernel의 (수정/추가 필요)

---

### 참고 자료들 
- https://github.com/joonhyung-lee/mujoco-robotics-usage
- https://github.com/stack-of-tasks/pinocchio
- https://github.com/ozkannceylan/mujoco-robotics-lab
- https://github.com/elijah-waichong-chan/go2-convex-mpc
- https://github.com/Shunichi09/PythonLinearNonlinearControl

---

### To Do Lists
1. Introduction to Humanoid Robotics - kajita ✅
2. Modern Robotics - Kevin M. Lynch, Frank C. Park ✅
3. Convex Optimization - Stephen Boyd - ~ing
4. Underactuated Robotics - Russ Tedrake
5. Convex Model Predictive Control - MIT 


---