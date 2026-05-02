import casadi as ca
import numpy as np
import scipy.sparese as sp
from core import pinocchio_wrapper
import time 

COST_MATRIX_Q = np.diag([1, 1, 50,  10, 20, 1,  2, 2, 1,  1, 1, 1])     # State cost weight matrix
COST_MATRIX_R = np.diag([1e-5] * 12) 

class Centroidal ():
    
    def __init__(self):
        
        self.A = A
        
    
    def 