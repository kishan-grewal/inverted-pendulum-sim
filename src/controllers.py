import numpy as np
import control

from dynamics import linearise


class LQRController:
    def __init__(self, Q=None, R=None, output_limits=None):
        A, B = linearise()
        
        # Default weights if not provided
        if Q is None:
            Q = np.diag([10.0, 1.0, 100.0, 10.0])  # [x, x_dot, theta, theta_dot]
        if R is None:
            R = np.array([[1.0]])  # control effort
        
        # Compute LQR gain
        K, S, E = control.lqr(A, B, Q, R)
        self.K = np.array(K).flatten()  # shape (4,)
        
        self.output_limits = output_limits
    
    def reset(self):
        pass  # no internal state
    
    def compute(self, state, dt):
        F = -self.K @ state
        
        if self.output_limits is not None:
            F = np.clip(F, self.output_limits[0], self.output_limits[1])
        
        return F