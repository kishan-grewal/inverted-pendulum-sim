import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.dynamics import get_parameters
from src.controllers import LQRController, PIDController, PolePlacementController
from src.observers import LuenbergerObserver, DEFAULT_OBSERVER_POLES
from src.simulation import simulate, DEFAULT_NOISE_STD_X, DEFAULT_NOISE_STD_THETA
from src.visualisation import animate_from_arrays
from src.plots import plot_time_series, plot_phase_portrait
from src.metrics import calculate_settling_time, filter_angle_signal


# ===== SYSTEM CONSTRAINTS =====
FORCE_LIMITS = (-15.0, 15.0)  # [N]

# ===== SIMULATION DURATIONS =====
A_DURATION = 5.0  # [s]
B_DURATION = 5.0  # [s]
C_DURATION = 5.0  # [s]

# ===== EVALUATION A DISTURBANCE =====
# Format: (time [s], cart_impulse [N·s], angular_impulse [N·s·m])
# Uncomment ONE of these:
EVAL_A_DISTURBANCE = (1.0, 0.0, 0.01)   # Small tap on pendulum
# EVAL_A_DISTURBANCE = (1.0, 0.0, 0.03)    # Large tap on pendulum  
# EVAL_A_DISTURBANCE = (1.0, 2.0, 0.0)    # Shove to cart

# ===== EVALUATION C PARAMETERS =====
EVAL_C_TARGET = 2.0  # [m] - target position for eval C
EVAL_C_TOLERANCE = 0.10  # [m] - within 10cm counts as "reached"


def create_controller(controller_type, poles=None):
    """Create controller instance based on type."""
    if controller_type == 'lqr':
        return LQRController(output_limits=FORCE_LIMITS)
    elif controller_type == 'pid':
        return PIDController(output_limits=FORCE_LIMITS)
    elif controller_type == 'pole':
        return PolePlacementController(poles=poles, output_limits=FORCE_LIMITS)


def save_results(filepath, t, states, control):
    """Save simulation results as simple CSV."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data = np.column_stack([t, states, control])
    np.savetxt(filepath, data, delimiter=',',
               header='t,x,x_dot,theta,theta_dot,F',
               comments='')


def run_evaluation(eval_type, test_id, controller_type, enable_air_drag=True,
                   show_textboxes=False, poles=None, use_observer=True):
    """
    Run evaluation A, B, or C.
    
    Args:
        eval_type: 'A', 'B', or 'C'
        test_id: For A: unused, For B: angle in degrees, For C: unused
        controller_type: 'lqr', 'pid', or 'pole'
        enable_air_drag: whether to include air drag
        show_textboxes: whether to show interactive textboxes
        poles: custom poles for pole placement controller
        use_observer: whether to use state observer
    """
    
    # Setup based on evaluation type
    if eval_type == 'A':
        disturbance = EVAL_A_DISTURBANCE
        disturbance_time = disturbance[0]
        initial_state = np.array([0.0, 0.0, 0.0, 0.0])
        disturbances = [disturbance]
        duration = A_DURATION
        eval_label = "A"
        target_state = None
        on_step = None
        xlim = None
        
    elif eval_type == 'B':
        theta_0 = np.radians(test_id)
        initial_state = np.array([0.0, 0.0, theta_0, 0.0])
        disturbances = None
        disturbance = None
        duration = B_DURATION
        eval_label = f"B ({test_id}°)"
        target_state = None
        on_step = None
        xlim = None
    
    elif eval_type == 'C':
        initial_state = np.array([0.0, 0.0, 0.0, 0.0])
        target_state = np.array([EVAL_C_TARGET, 0.0, 0.0, 0.0])
        disturbances = None
        disturbance = None
        duration = C_DURATION
        eval_label = "C (Sprint)"
        xlim = (-1, 3)  # Wider view for sprint
        
        # Setup controller first (needed for PID callback)
        controller = create_controller(controller_type, poles=poles)
        
        # PID special handling for sprint
        on_step = None
        if controller_type == 'pid':
            # More oscillatory gains for sprint (higher Kp, lower Kd)
            controller.set_gains(Kp=150.0, Ki=0.0, Kd=10.0)
            
            PID_LEAN_ANGLE = np.radians(1.0)   # lean forward
            PID_BRAKE_ANGLE = np.radians(-11.0)  # lean backward to brake
            
            def pid_sprint_callback(ctrl, state, t):
                x = state[0]
                
                if x < 1.7:
                    ctrl.target_angle = PID_LEAN_ANGLE
                elif x < 2.0:
                    ctrl.target_angle = PID_BRAKE_ANGLE
                else:
                    ctrl.target_angle = 0.0
            
            on_step = pid_sprint_callback
    
    print(f"Evaluation {eval_label}")
    print(f"Controller: {controller_type.upper()}")
    
    # Setup controller and observer (if not already created for PID sprint)
    t_span = (0.0, duration)
    if eval_type != 'C' or controller_type != 'pid':
        controller = create_controller(controller_type, poles=poles)
    observer = None
    if use_observer:
        observer = LuenbergerObserver(observer_poles=DEFAULT_OBSERVER_POLES)
    
    # Run simulation
    results = simulate(
        initial_state, t_span,
        controller=controller,
        enable_air_drag=enable_air_drag,
        observer=observer,
        noise_std_x=DEFAULT_NOISE_STD_X,
        noise_std_theta=DEFAULT_NOISE_STD_THETA,
        disturbances=disturbances,
        target_state=target_state,
        on_step=on_step
    )
    
    t = results['t']
    states = results['states']
    control = results['control']
    
    # Pad control array if needed
    if len(control) < len(states):
        control = np.concatenate([[0.0], control])
    
    # Calculate metrics
    if eval_type == 'A':
        dist_idx = np.argmin(np.abs(t - disturbance_time))
        theta_after = np.abs(np.degrees(states[dist_idx:, 2]))
        max_deviation = theta_after.max()
        
        # Settling time with filtered signal
        settling_time = calculate_settling_time(
            t, 
            np.degrees(states[:, 2]),
            initial_deviation=max_deviation,
            disturbance_time=disturbance_time
        )
        
        fell = np.any(theta_after > 90.0)
        
    elif eval_type == 'B':
        max_deviation = np.abs(test_id)
        
        # Filter signal for checks
        theta_deg_full = np.degrees(states[:, 2])
        theta_filtered = filter_angle_signal(t, theta_deg_full)
        
        # Check stabilization: final 1 second within 1°
        final_theta_filtered = np.abs(theta_filtered[-100:])
        stabilised = np.all(final_theta_filtered < 1.0)
        
        # Settling time
        settling_time = calculate_settling_time(
            t,
            theta_deg_full,
            initial_deviation=max_deviation,
            disturbance_time=0.0
        )
        
        fell = False
    
    elif eval_type == 'C':
        # Sprint metrics
        positions = states[:, 0]
        theta_deg = np.abs(np.degrees(states[:, 2]))
        
        # Time to reach target (within tolerance)
        reached_idx = np.where(np.abs(positions - EVAL_C_TARGET) < EVAL_C_TOLERANCE)[0]
        time_to_target = t[reached_idx[0]] if len(reached_idx) > 0 else None
        
        # Final position error
        final_position = positions[-1]
        final_error = abs(final_position - EVAL_C_TARGET)
        
        # Max angle deviation
        max_deviation = theta_deg.max()
        
        # Did it fall?
        fell = max_deviation > 90.0
        
        # Final velocity (should be near zero if stopped)
        final_velocity = abs(states[-1, 1])
    
    max_force = np.max(np.abs(control))
    
    # Print results
    print(f"\nResults:")
    
    if eval_type == 'A':
        print(f"  Max angle deviation: {max_deviation:.2f}°")
        
        if fell:
            print(f"  Status: FAILED - Pendulum fell over")
        elif settling_time:
            print(f"  Settling time: {settling_time:.2f}s")
            print(f"  Status: SUCCESS")
        else:
            print(f"  Settling time: Did not settle within {duration}s")
    
    elif eval_type == 'B':
        print(f"  Max angle deviation: {max_deviation:.2f}°")
        print(f"  Stabilised: {'Yes' if stabilised else 'No'}")
        if settling_time:
            print(f"  Settling time: {settling_time:.2f}s")
        else:
            print(f"  Settling time: Did not settle")
    
    elif eval_type == 'C':
        if fell:
            print(f"  Status: FAILED - Pendulum fell over")
        else:
            if time_to_target:
                print(f"  Time to reach {EVAL_C_TARGET}m: {time_to_target:.2f}s")
            else:
                print(f"  Time to reach {EVAL_C_TARGET}m: Did not reach target")
            print(f"  Final position: {final_position:.3f}m (error: {final_error:.3f}m)")
            print(f"  Final velocity: {final_velocity:.3f}m/s")
            print(f"  Max angle deviation: {max_deviation:.2f}°")
            if final_error < EVAL_C_TOLERANCE and final_velocity < 0.1:
                print(f"  Status: SUCCESS")
            else:
                print(f"  Status: PARTIAL - reached but not settled")
    
    print(f"  Max control force: {max_force:.2f} N\n")
    
    # Save results
    data_dir = Path(__file__).parent / "data"
    drag_str = "drag" if enable_air_drag else "nodrag"
    obs_str = "obs" if use_observer else "perfect"
    
    if eval_type == 'A':
        filename = f"eval_a_{controller_type}_{drag_str}_{obs_str}.csv"
    elif eval_type == 'B':
        filename = f"eval_b_{controller_type}_{drag_str}_{obs_str}_{abs(test_id)}deg.csv"
    elif eval_type == 'C':
        filename = f"eval_c_{controller_type}_{drag_str}_{obs_str}.csv"
    
    save_results(data_dir / filename, t, states, control)
    
    # Generate plots
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if eval_type == 'A':
        title_suffix = f" - Eval A ({controller_type.upper()})"
    elif eval_type == 'B':
        title_suffix = f" - Eval B: {test_id}° ({controller_type.upper()})"
    elif eval_type == 'C':
        title_suffix = f" - Eval C: Sprint ({controller_type.upper()})"
    
    fig1 = plot_time_series(t, states, control=control, disturbance=disturbance if eval_type == 'A' else None, title_suffix=title_suffix)
    fig1.savefig(plots_dir / filename.replace('.csv', '_timeseries.png'), dpi=150, bbox_inches='tight')
    
    fig2 = plot_phase_portrait(states, title_suffix=title_suffix)
    fig2.savefig(plots_dir / filename.replace('.csv', '_phase.png'), dpi=150, bbox_inches='tight')
    
    plt.close('all')
    
    # Show animation
    if eval_type == 'A':
        title = f"Eval A: {controller_type.upper()}"
    elif eval_type == 'B':
        title = f"Eval B: {controller_type.upper()} from {test_id}°"
    elif eval_type == 'C':
        title = f"Eval C: {controller_type.upper()} sprint to {EVAL_C_TARGET}m"
    
    if not enable_air_drag:
        title += " (no drag)"
    if use_observer:
        title += " (with observer)"
    
    animate_from_arrays(
        t, states, title=title,
        controller=controller if show_textboxes else None,
        observer=observer if show_textboxes else None,
        show_textboxes=show_textboxes,
        initial_state=initial_state,
        t_span=t_span,
        enable_air_drag=enable_air_drag,
        control=control,
        noise_std_x=DEFAULT_NOISE_STD_X,
        noise_std_theta=DEFAULT_NOISE_STD_THETA,
        disturbances=disturbances,
        xlim=xlim,
        interval=20, skip_frames=2
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Cart-Pendulum Control Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Evaluation selection
    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument('-A', action='store_true', help='Eval A: Disturbance rejection')
    eval_group.add_argument('-B', type=float, metavar='ANGLE', help='Eval B: Recovery from angle [deg]')
    eval_group.add_argument('-C', action='store_true', help='Eval C: Sprint to 2m target while balancing')
    
    # Controller selection
    parser.add_argument('--controller', type=str, required=True, choices=['lqr', 'pid', 'pole'],
                        help='Controller type')
    parser.add_argument('--poles', type=str, default=None,
                        help='Comma-separated poles for pole placement (e.g., "-2,-3,-4,-5")')
    
    # Feature flags
    parser.add_argument('--no-textboxes', action='store_true',
                        help='Disable interactive textboxes')
    parser.add_argument('--no-observer', action='store_true',
                        help='Use perfect state feedback instead of observer')
    parser.add_argument('--no-drag', action='store_true',
                        help='Disable air drag')
    
    args = parser.parse_args()

    # Parse poles if provided
    poles = None
    if args.poles is not None:
        poles = [complex(p.strip()) for p in args.poles.split(',')]
    
    # Determine evaluation type
    if args.A:
        eval_type = 'A'
        test_id = None
    elif args.B:
        eval_type = 'B'
        test_id = args.B
    elif args.C:
        eval_type = 'C'
        test_id = None

    # Run evaluation
    run_evaluation(
        eval_type=eval_type,
        test_id=test_id,
        controller_type=args.controller,
        enable_air_drag=not args.no_drag,
        show_textboxes=not args.no_textboxes,
        poles=poles,
        use_observer=not args.no_observer
    )


if __name__ == "__main__":
    main()