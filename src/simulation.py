import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from dynamics import state_derivative, get_parameters


def simulate(initial_state, t_span, dt=0.01, force_func=None):
    if force_func is None:
        force_func = lambda t, state: 0.0
    
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    def dynamics(t, state):
        F = force_func(t, state)
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
    
    return solution.t, solution.y.T


def save_results(filepath, t, states, params, initial_angle_deg):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("# Cart-Pendulum Passive Drop Simulation\n")
        f.write(f"# Initial angle: {initial_angle_deg} degrees\n")
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


def run_passive_drop_test(initial_angle_deg, output_file):
    theta_0 = np.radians(initial_angle_deg)
    initial_state = np.array([0.0, 0.0, theta_0, 0.0])
    t_span = (0.0, 10.0)
    
    print(f"Running passive drop test: theta_0 = {initial_angle_deg}°")
    t, states = simulate(initial_state, t_span)
    
    params = get_parameters()
    save_results(output_file, t, states, params, initial_angle_deg)
    print(f"Results saved to: {output_file}")
    
    print(f"  Initial state: x={initial_state[0]:.3f}m, theta={np.degrees(initial_state[2]):.1f}°")
    print(f"  Final state:   x={states[-1, 0]:.3f}m, theta={np.degrees(states[-1, 2]):.1f}°")
    print()


def main():
    data_dir = Path(__file__).parent.parent / "data"
    
    run_passive_drop_test(
        initial_angle_deg=10.0,
        output_file=data_dir / "passive_drop_positive_10deg.txt",
    )
    
    run_passive_drop_test(
        initial_angle_deg=-10.0,
        output_file=data_dir / "passive_drop_negative_10deg.txt",
    )


if __name__ == "__main__":
    main()