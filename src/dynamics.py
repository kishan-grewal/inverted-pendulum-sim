"""
Cart-pendulum dynamics module.

State vector: [x, x_dot, theta, theta_dot]
    x       - cart position [m]
    x_dot   - cart velocity [m/s]
    theta   - pendulum angle from vertical [rad], 0 = upright, positive = tilted right
    theta_dot - pendulum angular velocity [rad/s]

Input: F - horizontal force on cart [N], positive = rightward
"""

import numpy as np


# Physical parameters
# These match the spec: 60-100cm rod with 50g tip mass

M = 1.0             # cart mass [kg]
m_rod = 0.1         # rod mass [kg] (estimate for 12mm diameter wooden/aluminium rod)
m_tip = 0.05        # tip mass [kg] (specified as 50g)
m_pend = m_rod + m_tip  # total pendulum mass [kg]

L_rod = 0.8         # rod length [m] (80cm, middle of 60-100cm range)
r_tip = 0.01579     # tip disk radius [m] (31.58mm diameter)
h_tip = 0.0077      # tip disk thickness [m] (7.7mm)

# Centre of mass location from pivot
# Rod CoM at L_rod/2, tip at L_rod
l = (m_rod * L_rod / 2 + m_tip * L_rod) / m_pend  # distance from pivot to combined CoM [m]

# Moment of inertia about CoM
# Rod: (1/12) * m * L^2 about its own centre
# Tip: approximate as point mass at L_rod (disk inertia negligible for thin disk)
I_rod_com = (1 / 12) * m_rod * L_rod ** 2
I_tip_com = 0.0  # treating tip as point mass

# Parallel axis theorem to get inertia about pivot
d_rod = L_rod / 2  # distance from pivot to rod CoM
d_tip = L_rod      # distance from pivot to tip

I_rod_pivot = I_rod_com + m_rod * d_rod ** 2
I_tip_pivot = I_tip_com + m_tip * d_tip ** 2
I_pivot = I_rod_pivot + I_tip_pivot  # total moment of inertia about pivot [kg·m²]

# Damping coefficients
b_x = 0.1           # cart viscous friction [N·s/m]
b_theta = 0.001     # pivot viscous friction [N·m·s/rad] (must be low, spec requires zeta < 0.01)

# Gravity
g = 9.81            # [m/s²]

# Derived quantities
M_t = M + m_pend    # total translating mass
ml = m_pend * l     # mass-length product


def state_derivative(t: float, state: np.ndarray, F: float = 0.0) -> np.ndarray:
    """
    Compute state derivative for cart-pendulum system.
    
    Args:
        t: time [s] (unused, required for solve_ivp interface)
        state: [x, x_dot, theta, theta_dot]
        F: horizontal force on cart [N]
    
    Returns:
        [x_dot, x_ddot, theta_dot, theta_ddot]
    """
    x, x_dot, theta, theta_dot = state
    
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Mass matrix determinant
    D = M_t * I_pivot - (ml * c) ** 2
    
    # Right-hand side terms
    f1 = F - b_x * x_dot + ml * theta_dot ** 2 * s
    f2 = ml * g * s - b_theta * theta_dot
    
    # Explicit accelerations from matrix inverse
    x_ddot = (I_pivot * f1 - ml * c * f2) / D
    theta_ddot = (M_t * f2 - ml * c * f1) / D
    
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


def get_parameters() -> dict:
    """Return dictionary of all physical parameters for logging."""
    return {
        'M': M,
        'm_rod': m_rod,
        'm_tip': m_tip,
        'm_pend': m_pend,
        'L_rod': L_rod,
        'l': l,
        'I_pivot': I_pivot,
        'b_x': b_x,
        'b_theta': b_theta,
        'g': g,
        'M_t': M_t,
        'ml': ml,
    }