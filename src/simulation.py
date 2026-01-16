"""
Simulation module for cart-pendulum system.

Integrates dynamics using scipy.integrate.solve_ivp and provides
test scenarios for validating system behaviour.
"""

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from dynamics import state_derivative, get_parameters


def simulate(
    initial_state: np.ndarray,
    t_span: tuple[float, float],
    dt: float = 0.01,
    force_func: callable = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate cart-pendulum system.
    
    Args:
        initial_state: [x, x_dot, theta, theta_dot]
        t_span: (t_start, t_end) in seconds
        dt: time step for output sampling [s]
        force_func: callable(t, state) -> F, defaults to zero force
    
    Returns:
        t: time array
        states: state array, shape (n_steps, 4)
    """
    if force_func is None:
        force_func = lambda t, state: 0.0
    
    # Generate evaluation times
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    # Wrapper for solve_ivp that includes force
    def dynamics(t, state):
        F = force_func(t, state)
        return state_derivative(t, state, F)
    
    # Integrate
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


def save_results(
    filepath: Path,
    t: np.ndarray,
    states: np.ndarray,
    params: dict,
    initial_angle_deg: float,
):
    """
    Save simulation results to text file.
    
    Args:
        filepath: output file path
        t: time array
        states: state array, shape (n_steps, 4)
        params: physical parameters dictionary
        initial_angle_deg: initial angle in degrees (for header)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        # Header with parameters
        f.write("# Cart-Pendulum Passive Drop Simulation\n")
        f.write(f"# Initial angle: {initial_angle_deg} degrees\n")
        f.write("#\n")
        f.write("# Physical parameters:\n")
        for key, value in params.items():
            f.write(f"#   {key}: {value}\n")
        f.write("#\n")
        f.write("# Columns: t [s], x [m], x_dot [m/s], theta [rad], theta_dot [rad/s]\n")
        f.write("#\n")
        
        # Data
        for i in range(len(t)):
            f.write(f"{t[i]:.6f}\t{states[i, 0]:.6f}\t{states[i, 1]:.6f}\t"
                    f"{states[i, 2]:.6f}\t{states[i, 3]:.6f}\n")


def run_passive_drop_test(initial_angle_deg: float, output_file: str):
    """
    Run passive drop test from specified initial angle.
    
    Pendulum starts at given angle with zero velocity, no control force.
    System evolves under gravity and damping.
    
    Args:
        initial_angle_deg: initial pendulum angle [degrees], positive = tilted right
        output_file: path to save results
    """
    # Convert to radians
    theta_0 = np.radians(initial_angle_deg)
    
    # Initial state: cart at origin, at rest, pendulum at angle, at rest
    initial_state = np.array([0.0, 0.0, theta_0, 0.0])
    
    # Simulate for 10 seconds (enough to see settling)
    t_span = (0.0, 10.0)
    
    print(f"Running passive drop test: theta_0 = {initial_angle_deg}°")
    t, states = simulate(initial_state, t_span)
    
    # Save results
    params = get_parameters()
    save_results(output_file, t, states, params, initial_angle_deg)
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print(f"  Initial state: x={initial_state[0]:.3f}m, theta={np.degrees(initial_state[2]):.1f}°")
    print(f"  Final state:   x={states[-1, 0]:.3f}m, theta={np.degrees(states[-1, 2]):.1f}°")
    print()


def main():
    """Run both positive and negative angle drop tests."""
    output_dir = Path(__file__).parent.parent / "output"
    
    # Test 1: Drop from +10 degrees (tilted right)
    run_passive_drop_test(
        initial_angle_deg=10.0,
        output_file=output_dir / "passive_drop_positive_10deg.txt",
    )
    
    # Test 2: Drop from -10 degrees (tilted left)
    run_passive_drop_test(
        initial_angle_deg=-10.0,
        output_file=output_dir / "passive_drop_negative_10deg.txt",
    )


if __name__ == "__main__":
    main()