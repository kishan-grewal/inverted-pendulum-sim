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
            Q = np.diag([1.0, 1.0, 100.0, 1.0]) #1
        
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
    

class PolePlacementController:
    """
    Pole Placement (State Feedback) Controller for cart-pendulum stabilisation.

    Computes state feedback gain K by placing the closed-loop poles at
    user-specified locations. Unlike LQR which optimises a cost function,
    pole placement directly prescribes the system's dynamic response.

    For a stable system, all poles must have negative real parts.
    Poles further left = faster response but higher control effort.

    The closed-loop system is: x_dot = (A - B @ K) @ x
    where eigenvalues of (A - B @ K) are the desired poles.
    """

    def __init__(self, poles=None, output_limits=None):
        """
        Initialise pole placement controller.

        Args:
            poles: Desired closed-loop poles (4 values for 4th order system).
                   Can be real or complex conjugate pairs.
                   Default: [-2, -3, -4, -5] (stable, moderately fast)
            output_limits: (min, max) force limits in Newtons
        """
        A, B = linearise()

        # Default poles if not provided
        # These should all have negative real parts for stability
        # More negative = faster response but more aggressive control
        if poles is None:
            # Moderate response: settling time ~2-3 seconds
            poles = [-2.0, -3.0, -4.0, -5.0]

        self.desired_poles = np.array(poles)

        # Validate poles
        if len(self.desired_poles) != 4:
            raise ValueError(f"System is 4th order, need 4 poles, got {len(self.desired_poles)}")

        # Check stability (all poles should have negative real parts)
        if np.any(np.real(self.desired_poles) >= 0):
            raise ValueError("All poles must have negative real parts for stability")

        # Compute gain matrix K using pole placement
        # control.place() uses Ackermann's formula or similar methods
        K = control.place(A, B, self.desired_poles)

        self.K = np.array(K).flatten()  # shape (4,)
        self.A = A
        self.B = B
        self.output_limits = output_limits

        # Verify poles were placed correctly
        # A_cl = A - B @ K (K is 1x4 from control.place)
        A_cl = A - B @ K
        self.actual_poles = np.linalg.eigvals(A_cl)

    def reset(self):
        """Reset controller state (none for state feedback)."""
        pass

    def compute(self, state, dt):
        """
        Compute control force from state feedback.

        Args:
            state: [x, x_dot, theta, theta_dot]
            dt: time step (unused for pole placement)

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
            'type': 'PolePlacement',
            'K': self.K.tolist(),
            'desired_poles': self.desired_poles.tolist(),
            'actual_poles': self.actual_poles.tolist(),
        }

    @staticmethod
    def suggest_poles(settling_time=2.0, damping_ratio=0.7):
        """
        Suggest pole locations based on desired performance specs.

        For a second-order dominant response:
        - settling_time: approximate time to reach steady state (seconds)
        - damping_ratio: 0.7 is typical (underdamped but not too oscillatory)

        Returns 4 poles: 2 dominant (complex conjugates) + 2 fast real poles
        """
        # Dominant poles determine settling time
        # For 2% settling: ts ≈ 4 / (zeta * omega_n)
        # So omega_n ≈ 4 / (zeta * ts)
        zeta = damping_ratio
        omega_n = 4.0 / (zeta * settling_time)

        # Dominant complex conjugate poles
        sigma = -zeta * omega_n  # real part
        omega_d = omega_n * np.sqrt(1 - zeta**2)  # imaginary part

        dominant_poles = [
            complex(sigma, omega_d),
            complex(sigma, -omega_d)
        ]

        # Fast poles (3-5x further left, won't affect dominant response much)
        fast_poles = [
            3 * sigma,
            4 * sigma
        ]

        return dominant_poles + fast_poles