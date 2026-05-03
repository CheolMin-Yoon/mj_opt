import numpy as np
import pinocchio as pin

class TrajOptimizer():
    def __init__(self):
        pass
    
    
    '''
    
    출력:
      1. des_x_pos, des_y_pos
      2. des_x_vel, des_y_vel
      3. yaw_rate
    
    '''
    def compute_desired_command(self):
        return 0.0, 0.0