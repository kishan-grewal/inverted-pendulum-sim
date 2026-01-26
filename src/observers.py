import numpy as np
import control

from src.dynamics import linearise


class LuenbergerObserver:
    def __init__(self, observer_poles=None):
        A, B = linearise()
        
        # Measurement matrix: we measure x and theta
        C = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ])
        
        if observer_poles is None:
            observer_poles = [-10.0, -12.0, -15.0, -18.0]
        
        self.desired_poles = np.array(observer_poles)
        
        # Compute L using duality: poles of (A - LC) = poles of (A^T - C^T L^T)
        L_T = control.place(A.T, C.T, self.desired_poles)
        L = L_T.T
        
        self.A = A
        self.B = B
        self.C = C
        self.L = L
        
        self.x_hat = np.zeros(4)
        
        A_observer = A - L @ C
        self.actual_poles = np.linalg.eigvals(A_observer)
    
    def reset(self, initial_estimate=None):
        if initial_estimate is not None:
            self.x_hat = np.array(initial_estimate)
        else:
            self.x_hat = np.zeros(4)
    
    def update(self, y_measured, u, dt):
        y_pred = self.C @ self.x_hat
        innovation = y_measured - y_pred
        
        B_flat = self.B.flatten()
        x_hat_dot = self.A @ self.x_hat + B_flat * u + self.L @ innovation
        
        # Euler integration
        self.x_hat += x_hat_dot * dt
        
        return self.x_hat.copy()
    
    def get_estimate(self):
        return self.x_hat.copy()
    
    def get_info(self):
        return {
            'type': 'Luenberger',
            'L': self.L.tolist(),
            'desired_poles': self.desired_poles.tolist(),
            'actual_poles': self.actual_poles.tolist(),
        }