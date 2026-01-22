import numpy as np
import control

from src.dynamics import linearise


class LQRController:
    """
    Linear Quadratic Regulator for cart-pendulum stabilisation.
    
    Computes optimal state feedback gain K by solving the continuous-time
    algebraic Riccati equation. The gain is calculated based on the linearised
    system dynamics and user-specified Q (state cost) and R (control cost) matrices.
    
    The closed-loop poles are determined by the solution, not prescribed.
    """
    
    def __init__(self, Q=None, R=None, output_limits=None):
        A, B = linearise()
        
        # Default weights if not provided
        # Q penalises state deviation: [x, x_dot, theta, theta_dot]
        # Higher weight = more aggressive correction for that state
        if Q is None:
            Q = np.diag([1.0, 1.0, 100.0, 10.0])
        
        # R penalises control effort (scalar for single input)
        # Higher R = less aggressive control, slower response
        if R is None:
            R = np.array([[0.1]])
        
        # Solve continuous-time algebraic Riccati equation
        # Returns: K (gain), S (solution to ARE), E (closed-loop eigenvalues)
        K, S, E = control.lqr(A, B, Q, R)
        
        self.K = np.array(K).flatten()  # shape (4,)
        self.Q = Q
        self.R = R
        self.closed_loop_poles = E  # eigenvalues of (A - B @ K)
        
        self.output_limits = output_limits
    
    def reset(self):
        """Reset controller state (none for LQR)."""
        pass
    
    def compute(self, state, dt):
        """
        Compute control force from state feedback.
        
        Args:
            state: [x, x_dot, theta, theta_dot]
            dt: time step (unused for LQR)
        
        Returns:
            F: control force [N]
        """
        F = -self.K @ state
        
        if self.output_limits is not None:
            F = np.clip(F, self.output_limits[0], self.output_limits[1])
        
        return F
    
    def get_info(self):
        """Return controller parameters for logging."""
        return {
            'type': 'LQR',
            'K': self.K.tolist(),
            'Q_diag': np.diag(self.Q).tolist(),
            'R': self.R[0, 0],
            'closed_loop_poles': self.closed_loop_poles.tolist(),
        }


class PIDController:
    """
    PID controller for pendulum angle stabilisation.
    
    Controls cart force based on pendulum angle error (theta).
    Uses derivative of theta directly from state rather than numerical differentiation.
    Includes anti-windup via integral clamping.
    """
    
    def __init__(self, Kp=None, Ki=None, Kd=None, output_limits=None):
        # Default gains (can be tuned via sliders)
        self.Kp = Kp if Kp is not None else 50.0
        self.Ki = Ki if Ki is not None else 0.0
        self.Kd = Kd if Kd is not None else 3.0
        
        self.output_limits = output_limits
        
        # Internal state
        self.integral = 0.0
        self.integral_limit = 10.0  # anti-windup clamp
    
    def reset(self):
        """Reset integral accumulator."""
        self.integral = 0.0
    
    def set_gains(self, Kp=None, Ki=None, Kd=None):
        """Update gains (for interactive tuning)."""
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd
    
    def compute(self, state, dt):
        """
        Compute control force from PID on angle error.
        
        Args:
            state: [x, x_dot, theta, theta_dot]
            dt: time step [s]
        
        Returns:
            F: control force [N]
        """
        theta = state[2]
        theta_dot = state[3]
        
        # Error is angle from upright (theta=0 is upright)
        error = theta
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        I = self.Ki * self.integral
        
        # Derivative term (use theta_dot directly, more accurate than numerical diff)
        D = self.Kd * theta_dot
        
        # Total control force
        # Positive theta (tilted right) -> negative force (push left) to restore
        F = P + I + D
        
        if self.output_limits is not None:
            F = np.clip(F, self.output_limits[0], self.output_limits[1])
        
        return F
    
    def get_info(self):
        """Return controller parameters for logging."""
        return {
            'type': 'PID',
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd,
        }