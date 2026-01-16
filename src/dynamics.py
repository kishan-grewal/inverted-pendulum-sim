import numpy as np


# Physical parameters - match spec: 60-100cm rod with 50g tip mass

M = 1.0  # cart mass [kg]
m_rod = 0.1  # rod mass [kg]
m_tip = 0.05  # tip mass [kg] (specified as 50g)
m_pend = m_rod + m_tip  # total pendulum mass [kg]

L_rod = 0.8  # rod length [m] (80cm, middle of 60-100cm range)
r_tip = 0.01579  # tip disk radius [m] (31.58mm diameter)
h_tip = 0.0077  # tip disk thickness [m] (7.7mm)

# Centre of mass location from pivot (rod CoM at L_rod/2, tip at L_rod)
l = (m_rod * L_rod / 2 + m_tip * L_rod) / m_pend

# Moment of inertia about CoM
I_rod_com = (1 / 12) * m_rod * L_rod ** 2
I_tip_com = 0.0  # treating tip as point mass

# Parallel axis theorem to get inertia about pivot
d_rod = L_rod / 2
d_tip = L_rod

I_rod_pivot = I_rod_com + m_rod * d_rod ** 2
I_tip_pivot = I_tip_com + m_tip * d_tip ** 2
I_pivot = I_rod_pivot + I_tip_pivot  # total moment of inertia about pivot [kg·m²]

# Damping coefficients
b_x = 0.1  # cart viscous friction [N·s/m]
b_theta = 0.001  # pivot viscous friction [N·m·s/rad] (must be low, spec requires zeta < 0.01)

# Gravity
g = 9.81  # [m/s²]

# Derived quantities
M_t = M + m_pend  # total translating mass
ml = m_pend * l  # mass-length product


def state_derivative(t, state, F=0.0):
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


def get_parameters():
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