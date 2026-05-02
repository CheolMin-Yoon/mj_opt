#### Mujoco, Pinocchio 공부자료

## 순수 무조코 
ur5e_mj_dls.ipynb

## 순수 DLS + pinocchio
ur5e_mj2pin_pure_dls.ipynb

## PTP 궤적 quintic, cubic
ur5e_mj2pin_PTP.ipynb

## waypoints 궤적 cubic
ur5e_mj2pin_Spline.ipynb

## cubic + 6D control
ur5e_mj2pin_Spline_6D.ipynb

## IK를 QP로 변경
ur5e_mj2pin_ik_qp.ipynb

## DLS + RRT*
ur5e_mj2pin_Spline_6D_RRT.ipynb

## OCS impedance control
ur5e_mj2pin_impedance.ipynb

### How to Make New 
1. core, utils를 복사한다.
2. fixed base, floating base인지에 따라서 컨테이너 클래스를 선택 후 내부 슬라이싱 수정 
3. pinocchio wrapper의 FRAMES는 manipulation 대상이므로 알맞은 URDF LINK NAME으로 수정
4. init 부분의 SE3, SO3 부분 수정 
5. mujoco kernel의 

#################################################################################

### Mobile 

## LQR error dynamics 
husky_LQR_tracking.ipynb

## MPC (현재 작업 중)

#################################################################################

### Legged Robot

## G1 or Go2 (미정)
## Convex MPC or MPC - iLQR (미정)

#################################################################################

### 참고 자료들 
https://github.com/joonhyung-lee/mujoco-robotics-usage
https://github.com/stack-of-tasks/pinocchio
https://github.com/ozkannceylan/mujoco-robotics-lab
https://github.com/elijah-waichong-chan/go2-convex-mpc
https://github.com/Shunichi09/PythonLinearNonlinearControl


### To Do Lists
1. motion planner에 Raibert Heurstic 추가
2. Visualizer.py에 Convex hull 시각화함수 추가 -> Graham scan
3. gait, swing phase planner 추가
4. WBC 튜닝

### git 명령어들
#