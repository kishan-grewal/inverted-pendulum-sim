import numpy as np
import control

from src.dynamics import linearise


class LQRController:
    def __init__(self, Q=None, R=None, output_limits=None,
                 position_weight=1.0, velocity_weight=1.0,
                 angle_weight=1.0, angular_velocity_weight=1.0,
                 control_weight=1.0):
        from src.dynamics import get_parameters
        
        self.params = get_parameters()
        self.output_limits = output_limits
        
        self._tuning_weights = {
            'position_weight': position_weight,
            'velocity_weight': velocity_weight,
            'angle_weight': angle_weight,
            'angular_velocity_weight': angular_velocity_weight,
            'control_weight': control_weight,
        }
        
        self._recompute_gain()
    
    def _recompute_gain(self):
        A, B = linearise()
        
        M_t = self.params['M_t']
        m_pend = self.params['m_pend']
        l = self.params['l']
        I_pivot = self.params['I_pivot']
        g = self.params['g']
        
        position_weight = self._tuning_weights['position_weight']
        velocity_weight = self._tuning_weights['velocity_weight']
        angle_weight = self._tuning_weights['angle_weight']
        angular_velocity_weight = self._tuning_weights['angular_velocity_weight']
        control_weight = self._tuning_weights['control_weight']
        
        # Q weights from physical energy: KE_cart, PE_pendulum, KE_rotation
        q_x = (1.0 / l**2) * position_weight
        q_xdot = M_t * velocity_weight
        q_theta = (m_pend * g * l) * angle_weight
        q_thetadot = I_pivot * angular_velocity_weight
        
        Q = np.diag([q_x, q_xdot, q_theta, q_thetadot])
        
        F_ref = M_t * g
        r = (1.0 / F_ref**2) * control_weight
        R = np.array([[r]])
        
        K, S, E = control.lqr(A, B, Q, R)
        
        self.K = np.array(K).flatten()
        self.Q = Q
        self.R = R
        self.closed_loop_poles = E
        
        self._energy_weights = {
            'q_x': q_x,
            'q_xdot': q_xdot,
            'q_theta': q_theta,
            'q_thetadot': q_thetadot,
            'r': R[0, 0],
            'F_ref': F_ref,
        }
    
    def set_weights(self, position_weight=None, velocity_weight=None,
                    angle_weight=None, angular_velocity_weight=None,
                    control_weight=None):
        if position_weight is not None:
            self._tuning_weights['position_weight'] = position_weight
        if velocity_weight is not None:
            self._tuning_weights['velocity_weight'] = velocity_weight
        if angle_weight is not None:
            self._tuning_weights['angle_weight'] = angle_weight
        if angular_velocity_weight is not None:
            self._tuning_weights['angular_velocity_weight'] = angular_velocity_weight
        if control_weight is not None:
            self._tuning_weights['control_weight'] = control_weight
        
        self._recompute_gain()
    
    def reset(self):
        pass
    
    def compute(self, state, dt):
        F = -self.K @ state
        
        if self.output_limits is not None:
            F = np.clip(F, self.output_limits[0], self.output_limits[1])
        
        return F
    
    def get_info(self):
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
    def __init__(self, Kp=None, Ki=None, Kd=None, output_limits=None):
        self.Kp = Kp if Kp is not None else 50.0
        self.Ki = Ki if Ki is not None else 0.0
        self.Kd = Kd if Kd is not None else 3.0
        
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.integral_limit = 10.0
    
    def reset(self):
        self.integral = 0.0
    
    def set_gains(self, Kp=None, Ki=None, Kd=None):
        if Kp is not None:
            self.Kp = Kp
        if Ki is not None:
            self.Ki = Ki
        if Kd is not None:
            self.Kd = Kd
    
    def compute(self, state, dt):
        theta = state[2]
        theta_dot = state[3]
        
        error = theta
        
        P = self.Kp * error
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        I = self.Ki * self.integral
        
        D = self.Kd * theta_dot
        
        F = P + I + D
        
        if self.output_limits is not None:
            F = np.clip(F, self.output_limits[0], self.output_limits[1])
        
        return F
    
    def get_info(self):
        return {
            'type': 'PID',
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd,
        }
    

class PolePlacementController:
    def __init__(self, poles=None, output_limits=None):
        self.output_limits = output_limits
        
        if poles is None:
            poles = [-2.0, -3.0, -4.0, -5.0]
        
        self.desired_poles = np.array(poles)
        
        self._recompute_gain()
    
    def _recompute_gain(self):
        A, B = linearise()
        
        K = control.place(A, B, self.desired_poles)

        self.K = np.array(K).flatten()
        self.A = A
        self.B = B

        A_cl = A - B @ K
        self.actual_poles = np.linalg.eigvals(A_cl)
    
    def set_poles(self, pole1=None, pole2=None, pole3=None, pole4=None):
        poles_list = list(self.desired_poles)
        
        if pole1 is not None:
            poles_list[0] = pole1
        if pole2 is not None:
            poles_list[1] = pole2
        if pole3 is not None:
            poles_list[2] = pole3
        if pole4 is not None:
            poles_list[3] = pole4
        
        self.desired_poles = np.array(poles_list)
        self._recompute_gain()

    def reset(self):
        pass

    def compute(self, state, dt):
        F = -self.K @ state

        if self.output_limits is not None:
            F = np.clip(F, self.output_limits[0], self.output_limits[1])

        return F

    def get_info(self):
        return {
            'type': 'PolePlacement',
            'K': self.K.tolist(),
            'desired_poles': self.desired_poles.tolist(),
            'actual_poles': self.actual_poles.tolist(),
        }