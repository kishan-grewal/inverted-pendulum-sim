import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from dynamics import state_derivative, get_parameters


def simulate(initial_state, t_span, dt=0.01, controller=None, observer=None):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    control_history = []
    estimated_states = []
    
    # Reset observer with initial guess (assume we know initial position/angle, not velocities)
    if observer is not None:
        initial_estimate = np.array([initial_state[0], 0.0, initial_state[2], 0.0])
        observer.reset(initial_estimate)
    
    prev_t = t_span[0]
    
    def dynamics(t, state):
        nonlocal prev_t
        
        actual_dt = t - prev_t if t > prev_t else dt
        prev_t = t
        
        if controller is None:
            F = 0.0
            if observer is not None:
                measurement = np.array([state[0], state[2]])
                estimated = observer.update(measurement, F, actual_dt)
                estimated_states.append(estimated)
        else:
            if observer is not None:
                # Controller uses estimated state
                measurement = np.array([state[0], state[2]])
                estimated = observer.update(measurement, control_history[-1] if control_history else 0.0, actual_dt)
                estimated_states.append(estimated)
                F = controller.compute(estimated, actual_dt)
            else:
                # Controller uses true state
                F = controller.compute(state, actual_dt)
        
        control_history.append(F)
        return state_derivative(t, state, F)
    
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
    
    if observer is not None:
        results['estimated_states'] = np.array(estimated_states)
    
    return results


def save_results(filepath, t, states, params, metadata=None):
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