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

### Mobile Manipulator 

## 매니퓰레이터와 팔 분리 (순수 무조코)
husky_ur5e_base.ipynb

## 간단한 QP (무조코 + OSQP) 
husky_ur5e_wbc_qp.ipynb

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