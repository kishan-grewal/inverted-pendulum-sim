import numpy as np
import control

from src.dynamics import linearise


class LQRController:
    """
    Linear Quadratic Regulator for cart-pendulum stabilisation.
    
    Computes optimal state feedback gain K by solving the continuous-time
    algebraic Riccati equation. Q and R matrices are derived from physical
    energy considerations, ensuring weights scale naturally with system parameters.
    
    Energy-based weighting:
        - Cart kinetic energy: ½ M_t ẋ²
        - Pendulum potential energy: m_pend g l (1 - cos θ) ≈ ½ m_pend g l θ²
        - Pendulum rotational kinetic energy: ½ I_pivot θ̇²
        - Cart position: scaled by 1/l² for dimensional consistency
        - Control: scaled by (M_t g)² representing force to accelerate at 1g
    
    The cost functional becomes a weighted sum of physical energies,
    making the trade-offs interpretable and automatically scaled.
    """
    
    def __init__(self, Q=None, R=None, output_limits=None,
                 position_weight=1.0, velocity_weight=1.0,
                 angle_weight=1.0, angular_velocity_weight=1.0,
                 control_weight=1.0):
        """
        Initialise LQR controller with energy-based weights.
        
        Args:
            Q: State cost matrix (4x4). If None, calculated from energy scaling.
            R: Control cost matrix (1x1). If None, calculated from energy scaling.
            output_limits: (min, max) force limits [N]
            position_weight: Multiplier for cart position cost (default 1.0)
            velocity_weight: Multiplier for cart velocity cost (default 1.0)
            angle_weight: Multiplier for pendulum angle cost (default 1.0)
            angular_velocity_weight: Multiplier for angular velocity cost (default 1.0)
            control_weight: Multiplier for control effort cost (default 1.0)
                            Higher = less aggressive control, slower response
        """
        from src.dynamics import get_parameters
        
        A, B = linearise()
        params = get_parameters()
        
        self.output_limits = output_limits
        
        # Extract physical parameters
        M_t = params['M_t']         # total translating mass [kg]
        m_pend = params['m_pend']   # pendulum mass [kg]
        l = params['l']             # pivot to CoM distance [m]
        I_pivot = params['I_pivot'] # moment of inertia about pivot [kg·m²]
        g = params['g']             # gravity [m/s²]
        
        # Energy-based Q matrix derivation:
        #
        # Cart position: no natural energy term, use 1/l² for dimensional 
        # consistency (penalise displacement relative to pendulum length)
        q_x = (1.0 / l**2) * position_weight
        
        # Cart velocity: from kinetic energy ½ M_t ẋ²
        # Coefficient in quadratic form is M_t
        q_xdot = M_t * velocity_weight
        
        # Pendulum angle: from potential energy m g l (1 - cos θ) ≈ ½ m g l θ²
        # Coefficient in quadratic form is m_pend * g * l
        q_theta = (m_pend * g * l) * angle_weight
        
        # Angular velocity: from rotational kinetic energy ½ I_pivot θ̇²
        # Coefficient in quadratic form is I_pivot
        q_thetadot = I_pivot * angular_velocity_weight
        
        if Q is None:
            Q = np.diag([q_x, q_xdot, q_theta, q_thetadot])
        
        # Energy-based R matrix:
        # Scale by (M_t * g)² - the force to accelerate total mass at 1g
        # This makes control cost dimensionally consistent with state costs
        # control_weight > 1 means more conservative control (less force)
        F_ref = M_t * g  # reference force scale [N]
        
        if R is None:
            r = (1.0 / F_ref**2) * control_weight
            R = np.array([[r]])
        
        # Solve continuous-time algebraic Riccati equation
        K, S, E = control.lqr(A, B, Q, R)
        
        self.K = np.array(K).flatten()  # shape (4,)
        self.Q = Q
        self.R = R
        self.closed_loop_poles = E
        
        # Store computed weights for logging
        self._energy_weights = {
            'q_x': q_x,
            'q_xdot': q_xdot,
            'q_theta': q_theta,
            'q_thetadot': q_thetadot,
            'r': R[0, 0],
            'F_ref': F_ref,
        }
        
        self._tuning_weights = {
            'position_weight': position_weight,
            'velocity_weight': velocity_weight,
            'angle_weight': angle_weight,
            'angular_velocity_weight': angular_velocity_weight,
            'control_weight': control_weight,
        }
    
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
            'type': 'LQR (energy-weighted)',
            'K': self.K.tolist(),
            'Q_diag': np.diag(self.Q).tolist(),
            'R': self.R[0, 0],
            'closed_loop_poles': [complex(p).real for p in self.closed_loop_poles],
            'tuning_weights': self._tuning_weights,
            'energy_weights': self._energy_weights,
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
        # Positive theta (tilted right) -> positive force (push right to catch it)
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