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


# ===== SYSTEM CONSTRAINTS =====
FORCE_LIMITS = (-15.0, 15.0)  # [N]

# ===== SIMULATION DURATIONS =====
A_DURATION = 7.0  # [s] - longer to see settling after disturbance
B_DURATION = 5.0  # [s]
SPRINT_DURATION = 10.0  # [s]

# ===== SPRINT PARAMETERS =====
SPRINT_TARGET = 2.0  # [m] - target position for sprint
SPRINT_POSITION_TOLERANCE = 0.05  # [m] - within 5cm counts as "reached"

# ===== EVALUATION A DISTURBANCE =====
# Format: (time [s], cart_impulse [N·s], angular_impulse [N·s·m])
# Uncomment ONE of these:
EVAL_A_DISTURBANCE = (1.0, 0.0, 0.01)   # Small tap on pendulum
# EVAL_A_DISTURBANCE = (1.0, 0.0, 0.05)    # Large tap on pendulum  
# EVAL_A_DISTURBANCE = (1.0, 2.0, 0.0)    # Shove to cart


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
    Run evaluation A or B.
    
    Args:
        eval_type: 'A' or 'B'
        test_id: For A: unused, For B: angle in degrees
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
        
    else:  # eval_type == 'B'
        theta_0 = np.radians(test_id)
        initial_state = np.array([0.0, 0.0, theta_0, 0.0])
        disturbances = None
        disturbance = None
        duration = B_DURATION
        eval_label = f"B ({test_id}°)"
    
    print(f"Evaluation {eval_label}")
    print(f"Controller: {controller_type.upper()}")
    
    # Setup controller and observer
    t_span = (0.0, duration)
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
        disturbances=disturbances
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
        
        # Settling time: within ±1° for 0.5s
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
        
        # Settling time: within ±0.5° for 0.5s
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
    else:
        print(f"  Stabilised: {'Yes' if stabilised else 'No'}")
        if settling_time:
            print(f"  Settling time: {settling_time:.2f}s")
        else:
            print(f"  Settling time: Did not settle")
    
    print(f"  Max control force: {max_force:.2f} N\n")
    
    # Save results
    data_dir = Path(__file__).parent / "data"
    drag_str = "drag" if enable_air_drag else "nodrag"
    obs_str = "obs" if use_observer else "perfect"
    
    if eval_type == 'A':
        filename = f"eval_a_{controller_type}_{drag_str}_{obs_str}.csv"
    else:
        filename = f"eval_b_{controller_type}_{drag_str}_{obs_str}_{abs(test_id)}deg.csv"
    
    save_results(data_dir / filename, t, states, control)
    
    # Generate plots
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if eval_type == 'A':
        title_suffix = f" - Eval A ({controller_type.upper()})"
    else:
        title_suffix = f" - Eval B: {test_id}° ({controller_type.upper()})"
    
    fig1 = plot_time_series(t, states, control=control, disturbance=disturbance, title_suffix=title_suffix)
    fig1.savefig(plots_dir / filename.replace('.csv', '_timeseries.png'), dpi=150, bbox_inches='tight')
    
    fig2 = plot_phase_portrait(states, title_suffix=title_suffix)
    fig2.savefig(plots_dir / filename.replace('.csv', '_phase.png'), dpi=150, bbox_inches='tight')
    
    plt.close('all')
    
    # Show animation
    if eval_type == 'A':
        title = f"Eval A: {controller_type.upper()}"
    else:
        title = f"Eval B: {controller_type.upper()} from {test_id}°"
    
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
        interval=20, skip_frames=2
    )
    
    return results


def run_sprint(controller_type, enable_air_drag=True, show_textboxes=False,
               poles=None, use_observer=True):
    """
    Run sprint evaluation: stabilise pendulum and reach 2m target.
    """
    target_state = np.array([SPRINT_TARGET, 0.0, 0.0, 0.0])
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])
    t_span = (0.0, SPRINT_DURATION)

    print(f"Sprint Challenge")
    print(f"Controller: {controller_type.upper()}")
    print(f"Target: {SPRINT_TARGET}m")

    controller = create_controller(controller_type, poles=poles)
    observer = None
    if use_observer:
        observer = LuenbergerObserver(observer_poles=DEFAULT_OBSERVER_POLES)

    # PID sprint: lean forward (2°) until 1.8m, then straighten up
    on_step = None
    if controller_type == 'pid':
        # More oscillatory gains for sprint (higher Kp, lower Kd)
        controller.set_gains(Kp=150.0, Ki=0.0, Kd=10.0)

        PID_LEAN_ANGLE = np.radians(1.0)   # lean forward
        PID_BRAKE_ANGLE = np.radians(-11.0)  # lean backward to brake

        def pid_sprint_callback(ctrl, state, t):

            flag1 = False
            flag2 = False
            x = state[0]

            if x < 1.7 and flag1==False:
                ctrl.target_angle = PID_LEAN_ANGLE
                flag1=True
            elif x < 2.0 and flag2==False:
                ctrl.target_angle = PID_BRAKE_ANGLE
                flag2=True
            else:
                ctrl.target_angle = 0.0

        on_step = pid_sprint_callback
        print(f"PID gains: Kp={controller.Kp}, Ki={controller.Ki}, Kd={controller.Kd}")

    results = simulate(
        initial_state, t_span,
        controller=controller,
        enable_air_drag=enable_air_drag,
        observer=observer,
        noise_std_x=DEFAULT_NOISE_STD_X,
        noise_std_theta=DEFAULT_NOISE_STD_THETA,
        target_state=target_state,
        on_step=on_step
    )

    t = results['t']
    states = results['states']
    control = results['control']

    if len(control) < len(states):
        control = np.concatenate([[0.0], control])

    # Calculate sprint metrics
    positions = states[:, 0]
    theta_deg = np.abs(np.degrees(states[:, 2]))

    # Time to reach target (within tolerance)
    reached_idx = np.where(np.abs(positions - SPRINT_TARGET) < SPRINT_POSITION_TOLERANCE)[0]
    time_to_target = t[reached_idx[0]] if len(reached_idx) > 0 else None

    # Final position error
    final_position = positions[-1]
    final_error = abs(final_position - SPRINT_TARGET)

    # Max angle deviation
    max_angle = theta_deg.max()

    # Did it fall?
    fell = max_angle > 90.0

    # Final velocity (should be near zero if stopped)
    final_velocity = abs(states[-1, 1])

    # Max control force used
    max_force = np.max(np.abs(control))

    # Print results
    print(f"\nResults:")
    if fell:
        print(f"  Status: FAILED - Pendulum fell over")
    else:
        if time_to_target:
            print(f"  Time to reach {SPRINT_TARGET}m: {time_to_target:.2f}s")
        else:
            print(f"  Time to reach {SPRINT_TARGET}m: Did not reach target")
        print(f"  Final position: {final_position:.3f}m (error: {final_error:.3f}m)")
        print(f"  Final velocity: {final_velocity:.3f}m/s")
        print(f"  Max angle deviation: {max_angle:.2f}°")
        print(f"  Max control force: {max_force:.2f}N")
        if final_error < SPRINT_POSITION_TOLERANCE and final_velocity < 0.1:
            print(f"  Status: SUCCESS")
        else:
            print(f"  Status: PARTIAL - reached but not settled")

    # Save results
    data_dir = Path(__file__).parent / "data"
    drag_str = "drag" if enable_air_drag else "nodrag"
    obs_str = "obs" if use_observer else "perfect"
    filename = f"sprint_{controller_type}_{drag_str}_{obs_str}.csv"
    save_results(data_dir / filename, t, states, control)

    # Generate plots
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    title_suffix = f" - Sprint ({controller_type.upper()})"
    fig1 = plot_time_series(t, states, control=control, title_suffix=title_suffix)
    fig1.savefig(plots_dir / filename.replace('.csv', '_timeseries.png'), dpi=150, bbox_inches='tight')

    fig2 = plot_phase_portrait(states, title_suffix=title_suffix)
    fig2.savefig(plots_dir / filename.replace('.csv', '_phase.png'), dpi=150, bbox_inches='tight')

    plt.close('all')

    # Show animation
    title = f"Sprint: {controller_type.upper()} to {SPRINT_TARGET}m"
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
        xlim=(-1, 3),  # Sprint window: -1m to 3m
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
    eval_group.add_argument('--sprint', action='store_true', help='Sprint: Reach 2m target while balancing')
    
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
    
    # Determine evaluation type and run
    if args.sprint:
        run_sprint(
            controller_type=args.controller,
            enable_air_drag=not args.no_drag,
            show_textboxes=not args.no_textboxes,
            poles=poles,
            use_observer=not args.no_observer
        )
    else:
        if args.A:
            eval_type = 'A'
            test_id = None
        else:
            eval_type = 'B'
            test_id = args.B

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