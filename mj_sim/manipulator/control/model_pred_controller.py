'''
TDL
'''

import numpy as np
import pinocchio as pin
import casadi as ca


class MPController(self):
    def __init__(self, model, traj_generator, horizon=20, dt=0.05):
        self.model = model
        self.traj_generator = traj_generator
        self.horizon = horizon
        self.dt = dt
        
        # MPC 최적화 문제 설정
        pass
    
    def solve_mpc(self, current_state, target_state):
        pass