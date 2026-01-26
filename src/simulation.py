import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from src.dynamics import state_derivative, get_parameters


def add_sensor_noise(true_state, noise_std_x=0.002, noise_std_theta=0.005):
    x_true = true_state[0]
    theta_true = true_state[2]
    
    x_measured = x_true + np.random.normal(0, noise_std_x)
    theta_measured = theta_true + np.random.normal(0, noise_std_theta)
    
    return np.array([x_measured, theta_measured])


def simulate(initial_state, t_span, dt=0.01, controller=None, enable_air_drag=True,
             observer=None, noise_std_x=0.002, noise_std_theta=0.005):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    control_history = []
    estimate_history = []
    measurement_history = []
    
    if observer is not None:
        observer.reset(initial_estimate=initial_state)
    
    # Current control (updated at each timestep)
    current_control = [0.0]
    
    def dynamics(t, state):
        return state_derivative(t, state, current_control[0], enable_air_drag=enable_air_drag)
    
    # Step-by-step integration at each timestep
    state = initial_state.copy()
    states_list = [state.copy()]
    
    for i in range(1, len(t_eval)):
        t_prev = t_eval[i-1]
        t_curr = t_eval[i]
        dt_step = t_curr - t_prev
        
        # Update observer once per timestep (not inside RK45 stages)
        if observer is None:
            controller_state = state
        else:
            y_measured = add_sensor_noise(state, noise_std_x, noise_std_theta)
            measurement_history.append(y_measured)
            
            controller_state = observer.update(y_measured, current_control[0], dt_step)
            estimate_history.append(controller_state.copy())
        
        if controller is None:
            F = 0.0
        else:
            F = controller.compute(controller_state, dt_step)
        
        current_control[0] = F
        control_history.append(F)
        
        # Integrate with constant control
        sol = solve_ivp(
            dynamics,
            (t_prev, t_curr),
            state,
            method='RK45',
            dense_output=False,
            rtol=1e-8,
            atol=1e-10,
        )
        
        state = sol.y[:, -1]
        states_list.append(state.copy())
    
    results = {
        't': t_eval,
        'states': np.array(states_list),
        'control': np.array(control_history),
    }
    
    if observer is not None:
        results['estimates'] = np.array(estimate_history)
        results['measurements'] = np.array(measurement_history)
    
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