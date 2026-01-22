import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from src.dynamics import state_derivative, get_parameters


def simulate(initial_state, t_span, dt=0.01, controller=None, enable_air_drag=True):
    """
    Simulate cart-pendulum system.
    
    Args:
        initial_state: [x, x_dot, theta, theta_dot]
        t_span: (t_start, t_end)
        dt: time step for output [s]
        controller: controller object with compute(state, dt) method
        enable_air_drag: if True, include quadratic air drag on pendulum
    
    Returns:
        dict with keys: 't', 'states', 'control'
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    control_history = []
    
    # Track previous time for dt calculation in controller
    prev_t = t_span[0]
    prev_F = 0.0
    
    def dynamics(t, state):
        nonlocal prev_t, prev_F
        
        actual_dt = t - prev_t if t > prev_t else dt
        prev_t = t
        
        if controller is None:
            F = 0.0
        else:
            F = controller.compute(state, actual_dt)
        
        prev_F = F
        control_history.append(F)
        
        return state_derivative(t, state, F, enable_air_drag=enable_air_drag)
    
    solution = solve_ivp(
        dynamics,
        t_span,
        initial_state,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    
    results = {
        't': solution.t,
        'states': solution.y.T,
        'control': np.array(control_history),
    }
    
    return results


def save_results(filepath, t, states, params, metadata=None):
    """Save simulation results to text file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("# Cart-Pendulum Simulation\n")
        if metadata:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
        f.write("#\n")
        f.write("# Physical parameters:\n")
        for key, value in params.items():
            f.write(f"#   {key}: {value}\n")
        f.write("#\n")
        f.write("# Columns: t [s], x [m], x_dot [m/s], theta [rad], theta_dot [rad/s]\n")
        f.write("#\n")
        
        for i in range(len(t)):
            f.write(f"{t[i]:.6f}\t{states[i, 0]:.6f}\t{states[i, 1]:.6f}\t"
                    f"{states[i, 2]:.6f}\t{states[i, 3]:.6f}\n")