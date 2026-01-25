import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from src.dynamics import state_derivative, get_parameters


def add_sensor_noise(true_state, noise_std_x=0.002, noise_std_theta=0.005):
    """
    Add Gaussian noise to position measurements.
    
    Args:
        true_state: [x, x_dot, theta, theta_dot]
        noise_std_x: standard deviation of cart position noise [m]
        noise_std_theta: standard deviation of pendulum angle noise [rad]
    
    Returns:
        [x_measured, theta_measured]
    """
    x_true = true_state[0]
    theta_true = true_state[2]
    
    x_measured = x_true + np.random.normal(0, noise_std_x)
    theta_measured = theta_true + np.random.normal(0, noise_std_theta)
    
    return np.array([x_measured, theta_measured])


def simulate(initial_state, t_span, dt=0.01, controller=None, enable_air_drag=True,
             observer=None, noise_std_x=0.002, noise_std_theta=0.005):
    """
    Simulate cart-pendulum system.
    
    Args:
        initial_state: [x, x_dot, theta, theta_dot]
        t_span: (t_start, t_end)
        dt: time step for output [s]
        controller: controller object with compute(state, dt) method
        enable_air_drag: if True, include quadratic air drag on pendulum
        observer: observer object with update(y_measured, u, dt) method.
                  If None, controller sees true state directly.
                  If provided, controller sees estimated state from noisy measurements.
        noise_std_x: standard deviation of cart position noise [m]
        noise_std_theta: standard deviation of pendulum angle noise [rad]
    
    Returns:
        dict with keys: 't', 'states', 'control', 'estimates' (if observer used),
                       'measurements' (if observer used)
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    control_history = []
    estimate_history = []
    measurement_history = []
    
    # Initialize observer if provided
    if observer is not None:
        observer.reset(initial_estimate=initial_state)
    
    # Current control input (updated at each t_eval point)
    current_control = [0.0]  # Use list to allow modification in nested function
    
    def dynamics(t, state):
        """Continuous dynamics - called many times per timestep by RK45."""
        return state_derivative(t, state, current_control[0], enable_air_drag=enable_air_drag)
    
    # Simulate step-by-step at each t_eval point
    state = initial_state.copy()
    states_list = [state.copy()]
    
    for i in range(1, len(t_eval)):
        t_prev = t_eval[i-1]
        t_curr = t_eval[i]
        dt_step = t_curr - t_prev
        
        # Get measurements and update observer (once per timestep)
        if observer is None:
            # Perfect sensing: controller sees true state
            controller_state = state
        else:
            # Noisy sensing: add noise to measurements, use observer
            y_measured = add_sensor_noise(state, noise_std_x, noise_std_theta)
            measurement_history.append(y_measured)
            
            # Observer estimates state from noisy measurements
            controller_state = observer.update(y_measured, current_control[0], dt_step)
            estimate_history.append(controller_state.copy())
        
        # Controller computes control input for this timestep
        if controller is None:
            F = 0.0
        else:
            F = controller.compute(controller_state, dt_step)
        
        current_control[0] = F
        control_history.append(F)
        
        # Integrate dynamics from t_prev to t_curr with constant control
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
    
    # Add observer-related data if observer was used
    if observer is not None:
        results['estimates'] = np.array(estimate_history)
        results['measurements'] = np.array(measurement_history)
    
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