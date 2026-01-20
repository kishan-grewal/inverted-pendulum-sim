import numpy as np
from scipy.signal import place_poles

from src.dynamics import linearise


class LuenbergerObserver:
    def __init__(self, poles=None):
        A, B = linearise()
        self.A = A
        self.B = B
        
        self.C = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ])
        
        if poles is None:
            poles = np.array([-10, -15, -20, -25])
        
        result = place_poles(A.T, self.C.T, poles)
        self.L = result.gain_matrix.T
        
        self.x_hat = np.zeros(4)
    
    def reset(self, initial_estimate=None):
        if initial_estimate is not None:
            self.x_hat = np.array(initial_estimate)
        else:
            self.x_hat = np.zeros(4)
    
    def update(self, measurement, u, dt):
        y = np.array(measurement)
        y_hat = self.C @ self.x_hat
        
        x_hat_dot = self.A @ self.x_hat + self.B.flatten() * u + self.L @ (y - y_hat)
        self.x_hat = self.x_hat + x_hat_dot * dt
        
        return self.x_hat.copy()
    
    def get_estimate(self):
        return self.x_hat.copy()