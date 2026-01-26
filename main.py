import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.dynamics import get_parameters
from src.controllers import LQRController, PIDController, PolePlacementController
from src.observers import LuenbergerObserver
from src.simulation import simulate, DEFAULT_NOISE_STD_X, DEFAULT_NOISE_STD_THETA
from src.visualisation import animate_from_arrays, plot_time_series, plot_phase_portrait


# ===== SYSTEM CONSTRAINTS =====
FORCE_LIMITS = (-15.0, 15.0)  # [N]

# ===== SIMULATION DEFAULTS =====
DEFAULT_DURATION = 5.0  # [s]

# ===== EVALUATION A DISTURBANCES =====
# Format: (time [s], cart_impulse [N·s], angular_impulse [N·s·m])
EVAL_A1_DISTURBANCE = (1.0, 0.0, 0.01)   # Small tap on pendulum
EVAL_A2_DISTURBANCE = (1.0, 0.0, 0.05)    # Large tap on pendulum  
EVAL_A3_DISTURBANCE = (1.0, 2.0, 0.0)    # Shove to cart

EVAL_A_DURATION = 7.0  # [s] - longer to see settling


def create_controller(controller_type, poles=None):
    if controller_type == 'lqr':
        return LQRController(output_limits=FORCE_LIMITS)
    elif controller_type == 'pid':
        return PIDController(output_limits=FORCE_LIMITS)
    elif controller_type == 'pole':
        return PolePlacementController(poles=poles, output_limits=FORCE_LIMITS)


def save_results(filepath, t, states, control, eval_name, settling_time, max_deviation, max_force):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(f"# Evaluation: {eval_name}\n")
        f.write(f"# Settling time: {settling_time}\n")
        f.write(f"# Max deviation: {max_deviation:.2f}°\n")
        f.write(f"# Max force: {max_force:.2f} N\n")
        f.write("#\n")
        f.write("# Columns: t [s], x [m], x_dot [m/s], theta [rad], theta_dot [rad/s], F [N]\n")
        f.write("#\n")
        
        for i in range(len(t)):
            f.write(f"{t[i]:.6f}\t{states[i, 0]:.6f}\t{states[i, 1]:.6f}\t"
                    f"{states[i, 2]:.6f}\t{states[i, 3]:.6f}\t{control[i]:.6f}\n")


def run_evaluation(eval_type, test_id, controller_type, duration, enable_air_drag=True,
                   show_animation=True, show_sliders=False, save=True, poles=None,
                   use_observer=False, observer_poles=None, 
                   noise_std_x=DEFAULT_NOISE_STD_X,
                   noise_std_theta=DEFAULT_NOISE_STD_THETA):
    """
    Unified evaluation runner for both A and B.
    
    Args:
        eval_type: 'A' or 'B'
        test_id: For A: 1/2/3 (test number), For B: angle in degrees
        controller_type: 'lqr', 'pid', or 'pole'
        duration: simulation duration [s]
    """
    
    # Setup based on evaluation type
    if eval_type == 'A':
        test_configs = {
            1: ("Small tap on pendulum", EVAL_A1_DISTURBANCE),
            2: ("Large tap on pendulum", EVAL_A2_DISTURBANCE),
            3: ("Shove to cart chassis", EVAL_A3_DISTURBANCE),
        }
        if test_id not in test_configs:
            raise ValueError(f"Invalid test number: {test_id}. Use 1, 2, or 3.")
        
        test_name, disturbance = test_configs[test_id]
        disturbance_time = disturbance[0]
        initial_state = np.array([0.0, 0.0, 0.0, 0.0])
        disturbances = [disturbance]
        eval_label = f"A{test_id}"
        
    else:  # eval_type == 'B'
        test_name = f"Recovery from {test_id}°"
        theta_0 = np.radians(test_id)
        initial_state = np.array([0.0, 0.0, theta_0, 0.0])
        disturbances = None
        disturbance = None
        eval_label = "B"
    
    print(f"Evaluation {eval_label}: {test_name}")
    print(f"Controller: {controller_type.upper()}")
    
    # Setup controller and observer
    t_span = (0.0, duration)
    controller = create_controller(controller_type, poles=poles)
    observer = None
    if use_observer:
        observer = LuenbergerObserver(observer_poles=observer_poles)
    
    # Run simulation
    results = simulate(
        initial_state, t_span,
        controller=controller,
        enable_air_drag=enable_air_drag,
        observer=observer,
        noise_std_x=noise_std_x,
        noise_std_theta=noise_std_theta,
        disturbances=disturbances
    )
    
    t = results['t']
    states = results['states']
    control = results['control']
    
    # Pad control
    if len(control) < len(states):
        control = np.concatenate([[0.0], control])
    
    # Calculate metrics based on evaluation type
    if eval_type == 'A':
        # Find disturbance index
        dist_idx = np.argmin(np.abs(t - disturbance_time))
        
        # Analyze response after disturbance
        theta_after = np.abs(np.degrees(states[dist_idx:, 2]))
        max_deviation = theta_after.max()
        
        # Find settling time (angle within ±1° for 0.5s = 50 samples)
        settling_idx = None
        for i in range(len(theta_after) - 50):
            if np.all(theta_after[i:i+50] < 1.0):
                settling_idx = i
                break
        
        settling_time = t[dist_idx + settling_idx] - disturbance_time if settling_idx else None
        fell = np.any(theta_after > 90.0)
        
    else:  # eval_type == 'B'
        max_deviation = np.abs(test_id)
        
        # Check stabilization
        final_theta_deg = np.abs(np.degrees(states[-100:, 2]))
        stabilised = np.all(final_theta_deg < 1.0)
        
        # Find settling time (angle within ±0.5° for 0.5s = 50 samples)
        theta_deg = np.abs(np.degrees(states[:, 2]))
        settling_idx = None
        for i in range(len(theta_deg) - 50):
            if np.all(theta_deg[i:i+50] < 0.5):
                settling_idx = i
                break
        
        settling_time = t[settling_idx] if settling_idx else None
        fell = False
    
    max_force = np.max(np.abs(control))
    
    # Print results
    print(f"\nResults:")
    print(f"  Max angle deviation: {max_deviation:.2f}°")
    
    if eval_type == 'A':
        if fell:
            print(f"  Status: FAILED - Pendulum fell over")
        elif settling_time:
            print(f"  Settling time: {settling_time:.2f}s")
            print(f"  Status: SUCCESS")
        else:
            print(f"  Settling time: Did not settle within {duration}s")
    else:  # eval_type == 'B'
        print(f"  Stabilised: {'Yes' if stabilised else 'No'}")
        if settling_time:
            print(f"  Settling time: {settling_time:.2f}s")
        else:
            print(f"  Settling time: Did not settle")
    
    print(f"  Max control force: {max_force:.2f} N\n")
    
    # Save results
    if save:
        data_dir = Path(__file__).parent / "data"
        drag_str = "drag" if enable_air_drag else "nodrag"
        obs_str = "obs" if use_observer else "perfect"
        
        if eval_type == 'A':
            filename = f"eval_a{test_id}_{controller_type}_{drag_str}_{obs_str}.txt"
            eval_name = f"A{test_id}: {test_name} ({controller_type.upper()})"
        else:
            filename = f"eval_b_{controller_type}_{drag_str}_{obs_str}_{abs(test_id)}deg.txt"
            eval_name = f"B: {test_name} ({controller_type.upper()})"
        
        settling_str = f"{settling_time:.2f}s" if settling_time else "N/A"
        save_results(data_dir / filename, t, states, control, eval_name, settling_str, max_deviation, max_force)
        
        # Generate plots
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        if eval_type == 'A':
            title_suffix = f" - A{test_id}: {test_name} ({controller_type.upper()})"
        else:
            title_suffix = f" - B: {test_name} ({controller_type.upper()})"
        
        fig1 = plot_time_series(t, states, control=control, disturbance=disturbance, title_suffix=title_suffix)
        fig1.savefig(plots_dir / filename.replace('.txt', '_timeseries.png'), dpi=150, bbox_inches='tight')
        
        fig2 = plot_phase_portrait(states, title_suffix=title_suffix)
        fig2.savefig(plots_dir / filename.replace('.txt', '_phase.png'), dpi=150, bbox_inches='tight')
        
        plt.close('all')
    
    # Show animation
    if show_animation:
        if eval_type == 'A':
            title = f"Eval A{test_id}: {controller_type.upper()} - {test_name}"
        else:
            title = f"Eval B: {controller_type.upper()} from {test_id}°"
        
        if not enable_air_drag:
            title += " (no drag)"
        if use_observer:
            title += " (with observer)"
        
        animate_from_arrays(
            t, states, title=title,
            controller=controller if show_sliders else None,
            observer=observer if show_sliders else None,
            show_sliders=show_sliders,
            initial_state=initial_state,
            t_span=t_span,
            enable_air_drag=enable_air_drag,
            control=control,
            noise_std_x=noise_std_x,
            noise_std_theta=noise_std_theta,
            disturbances=disturbances,
            interval=20, skip_frames=2
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Cart-Pendulum Control Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Mutually exclusive evaluation selection
    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument('-A1', action='store_true', help='Eval A1: Small tap on pendulum')
    eval_group.add_argument('-A2', action='store_true', help='Eval A2: Large tap on pendulum')
    eval_group.add_argument('-A3', action='store_true', help='Eval A3: Shove to cart')
    eval_group.add_argument('-B', type=float, metavar='ANGLE', help='Eval B: Recovery from angle')
    
    # Controller arguments
    parser.add_argument('--controller', type=str, required=True, choices=['lqr', 'pid', 'pole'])
    parser.add_argument('--poles', type=str, default=None)
    parser.add_argument('--textboxes', action='store_false')

    # Observer arguments
    parser.add_argument('--observer', action='store_false')
    parser.add_argument('--observer-poles', type=str, default=None)

    # Noise arguments
    parser.add_argument('--noise-x', type=float, default=DEFAULT_NOISE_STD_X, help='Position noise [m]')
    parser.add_argument('--noise-theta', type=float, default=DEFAULT_NOISE_STD_THETA, help='Angle noise [deg]')

    # Other arguments
    parser.add_argument('--duration', type=float, default=None, help='Simulation duration [s]')
    parser.add_argument('--no-drag', action='store_true')
    parser.add_argument('--no-animation', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    
    args = parser.parse_args()

    poles = None
    if args.poles is not None:
        poles = [complex(p.strip()) for p in args.poles.split(',')]
    
    observer_poles = None
    if args.observer_poles is not None:
        observer_poles = [complex(p.strip()) for p in args.observer_poles.split(',')]
    
    # Set duration and determine eval type
    if args.A1 or args.A2 or args.A3:
        duration = args.duration if args.duration is not None else EVAL_A_DURATION
        eval_type = 'A'
        test_id = 1 if args.A1 else (2 if args.A2 else 3)
    else:
        duration = args.duration if args.duration is not None else DEFAULT_DURATION
        eval_type = 'B'
        test_id = args.B

    # Common kwargs for all evaluations
    common_kwargs = {
        'controller_type': args.controller,
        'duration': duration,
        'enable_air_drag': not args.no_drag,
        'show_animation': not args.no_animation,
        'show_sliders': args.textboxes,
        'save': not args.no_save,
        'poles': poles,
        'use_observer': args.observer,
        'observer_poles': observer_poles,
        'noise_std_x': args.noise_x,
        'noise_std_theta': args.noise_theta,
    }

    run_evaluation(eval_type, test_id, **common_kwargs)


if __name__ == "__main__":
    main()