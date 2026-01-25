import numpy as np
import control

from src.dynamics import linearise


class LuenbergerObserver:
    """
    Luenberger state observer for cart-pendulum system.
    
    Estimates full state [x, x_dot, theta, theta_dot] from noisy measurements
    of position [x, theta] only.
    
    Observer dynamics:
        x_hat_dot = A @ x_hat + B @ u + L @ (y - C @ x_hat)
    
    where L is the observer gain computed via pole placement.
    Observer poles should be 3-5x faster than controller poles for good tracking
    without excessive noise amplification.
    """
    
    def __init__(self, observer_poles=None):
        """
        Initialize Luenberger observer.
        
        Args:
            observer_poles: Desired observer poles (4 values).
                           Default: [-10, -12, -15, -18] (5x faster than typical controller)
                           More negative = faster tracking but more noise sensitivity
        """
        A, B = linearise()
        
        # Measurement matrix: we can measure x and theta
        C = np.array([
            [1, 0, 0, 0],  # x measurement
            [0, 0, 1, 0],  # theta measurement
        ])
        
        # Default observer poles (should be faster than controller poles)
        if observer_poles is None:
            observer_poles = [-10.0, -12.0, -15.0, -18.0]
        
        self.desired_poles = np.array(observer_poles)
        
        # Validate poles
        if len(self.desired_poles) != 4:
            raise ValueError(f"Need 4 observer poles, got {len(self.desired_poles)}")
        
        if np.any(np.real(self.desired_poles) >= 0):
            raise ValueError("All observer poles must have negative real parts")
        
        # Compute observer gain L using duality
        # We want eigenvalues of (A - LC), which by duality is equivalent to
        # eigenvalues of (A^T - C^T L^T)
        # So we use pole placement on the dual system
        L_T = control.place(A.T, C.T, self.desired_poles)
        L = L_T.T
        
        self.A = A
        self.B = B
        self.C = C
        self.L = L
        
        # State estimate
        self.x_hat = np.zeros(4)
        
        # Verify poles
        A_observer = A - L @ C
        self.actual_poles = np.linalg.eigvals(A_observer)
    
    def reset(self, initial_estimate=None):
        """
        Reset observer state.
        
        Args:
            initial_estimate: Initial state estimate. If None, uses zeros.
        """
        if initial_estimate is not None:
            self.x_hat = np.array(initial_estimate)
        else:
            self.x_hat = np.zeros(4)
    
    def update(self, y_measured, u, dt):
        """
        Update state estimate based on measurements and control input.
        
        Args:
            y_measured: [x_measured, theta_measured] - noisy position measurements
            u: control input (force) [N]
            dt: time step [s]
        
        Returns:
            x_hat: estimated state [x, x_dot, theta, theta_dot]
        """
        # Innovation: difference between measured and predicted output
        y_pred = self.C @ self.x_hat
        innovation = y_measured - y_pred
        
        # Observer dynamics
        # x_hat_dot = A @ x_hat + B @ u + L @ innovation
        B_flat = self.B.flatten()
        x_hat_dot = self.A @ self.x_hat + B_flat * u + self.L @ innovation
        
        # Integrate (Euler method)
        # For better accuracy could use RK4, but Euler is fine for small dt
        self.x_hat += x_hat_dot * dt
        
        return self.x_hat.copy()
    
    def get_estimate(self):
        """Return current state estimate without updating."""
        return self.x_hat.copy()
    
    def get_info(self):
        """Return observer parameters for logging."""
        return {
            'type': 'Luenberger',
            'L': self.L.tolist(),
            'desired_poles': self.desired_poles.tolist(),
            'actual_poles': self.actual_poles.tolist(),
        }