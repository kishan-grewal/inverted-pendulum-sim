import numpy as np

MOTOR_MASS = 95
BRACKET_MASS = 8.5
NUCLEO_MASS = 78
ROD_MASS = 22.5
TIP_MASS = 50
BOARD_MASS = 190

CART_MASS = 4 * MOTOR_MASS + 4 * BRACKET_MASS + NUCLEO_MASS + BOARD_MASS  # total cart mass [g]

# Physical parameters - match spec: 60-100cm rod with 50g tip mass

M = CART_MASS / 1000  # cart mass [kg]
m_rod = ROD_MASS / 1000  # rod mass [kg]
m_tip = TIP_MASS / 1000  # tip mass [kg] (specified as 50g)
m_pend = m_rod + m_tip  # total pendulum mass [kg]

L_rod = 0.6  # rod length [m] (80cm, middle of 60-100cm range)

# double check these!!
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

# Air drag coefficient for pendulum
# Drag torque = -c_drag * theta_dot * |theta_dot| (quadratic in angular velocity)
# Approximated as thin rod in air: c_drag ~ 0.5 * rho * Cd * A * l^2
# With rho=1.2 kg/m³, Cd~1.2 for cylinder, A~0.01*0.8 m² (rod width * length)
c_drag = 0.002  # air drag coefficient [N·m·s²/rad²]

# Gravity
g = 9.81  # [m/s²]

# Derived quantities
M_t = M + m_pend  # total translating mass
ml = m_pend * l  # mass-length product


def state_derivative(t, state, F=0.0, enable_air_drag=True):
    """
    Compute state derivatives for cart-pendulum system.
    
    Args:
        t: time (unused, required by ODE solvers)
        state: [x, x_dot, theta, theta_dot]
        F: horizontal force on cart [N]
        enable_air_drag: if True, include quadratic air drag on pendulum
    
    Returns:
        [x_dot, x_ddot, theta_dot, theta_ddot]
    """
    x, x_dot, theta, theta_dot = state
    
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Mass matrix determinant
    D = M_t * I_pivot - (ml * c) ** 2
    
    # Air drag torque (quadratic, opposes motion)
    if enable_air_drag:
        tau_drag = -c_drag * theta_dot * np.abs(theta_dot)
    else:
        tau_drag = 0.0
    
    # Right-hand side terms
    f1 = F - b_x * x_dot + ml * theta_dot ** 2 * s
    f2 = ml * g * s - b_theta * theta_dot + tau_drag
    
    # Explicit accelerations from matrix inverse
    x_ddot = (I_pivot * f1 - ml * c * f2) / D
    theta_ddot = (M_t * f2 - ml * c * f1) / D
    
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


def linearise():
    """
    Linearise dynamics about upright equilibrium (theta=0, all velocities=0).
    Returns A, B matrices for state-space form: x_dot = A @ x + B @ u
    
    State: [x, x_dot, theta, theta_dot]
    Input: F (horizontal force)
    
    At theta=0: cos(theta)=1, sin(theta)=theta (small angle), theta_dot^2 terms vanish
    Air drag is quadratic so linearises to zero at equilibrium.
    """
    # At equilibrium: theta=0, so c=1, s=0, theta_dot=0
    # Mass matrix determinant at theta=0
    D0 = M_t * I_pivot - ml ** 2
    
    # Linearised equations (derived from Jacobian of state_derivative):
    #
    # x_ddot = (I_pivot / D0) * (F - b_x * x_dot) - (ml / D0) * (ml * g * theta - b_theta * theta_dot)
    # theta_ddot = (M_t / D0) * (ml * g * theta - b_theta * theta_dot) - (ml / D0) * (F - b_x * x_dot)
    #
    # Rearranging into A, B form:
    
    A = np.array([
        [0, 1, 0, 0],
        [0, -I_pivot * b_x / D0, -ml ** 2 * g / D0, ml * b_theta / D0],
        [0, 0, 0, 1],
        [0, ml * b_x / D0, M_t * ml * g / D0, -M_t * b_theta / D0],
    ])
    
    B = np.array([
        [0],
        [I_pivot / D0],
        [0],
        [-ml / D0],
    ])
    
    return A, B


def get_parameters():
    """Return dictionary of all physical parameters."""
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
        'c_drag': c_drag,
        'g': g,
        'M_t': M_t,
        'ml': ml,
    }