import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.dynamics import get_parameters
from src.controllers import LQRController, PIDController, PolePlacementController
from src.observers import LuenbergerObserver
from src.simulation import simulate, save_results, DEFAULT_NOISE_STD_X, DEFAULT_NOISE_STD_THETA
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


def run_eval_a(test_number, controller_type, duration=EVAL_A_DURATION, enable_air_drag=True,
               show_animation=True, show_sliders=False, save=True, poles=None,
               use_observer=False, observer_poles=None, 
               noise_std_x=DEFAULT_NOISE_STD_X,
               noise_std_theta=DEFAULT_NOISE_STD_THETA):
    """
    Run Evaluation A: Disturbance rejection from stable equilibrium.
    
    Args:
        test_number: 1 (small tap), 2 (large tap), or 3 (cart shove)
        controller_type: 'lqr', 'pid', or 'pole'
        duration: simulation duration [s]
    """
    test_configs = {
        1: ("Small tap on pendulum", EVAL_A1_DISTURBANCE),
        2: ("Large tap on pendulum", EVAL_A2_DISTURBANCE),
        3: ("Shove to cart chassis", EVAL_A3_DISTURBANCE),
    }
    
    if test_number not in test_configs:
        raise ValueError(f"Invalid test number: {test_number}. Use 1, 2, or 3.")
    
    test_name, disturbance = test_configs[test_number]
    disturbance_time = disturbance[0]
    
    print(f"Evaluation A{test_number}: {test_name}")
    print(f"  Controller: {controller_type.upper()}")
    print(f"  Disturbance at t={disturbance_time}s: cart={disturbance[1]}N·s, angular={disturbance[2]}N·s·m")
    print(f"  Air drag: {'enabled' if enable_air_drag else 'disabled'}")
    print(f"  Observer: {'enabled' if use_observer else 'disabled'}")
    if use_observer:
        print(f"  Noise: σ_x={noise_std_x:.4f}m, σ_θ={noise_std_theta:.2f}°")
    print("-" * 50)
    
    # Start at stable equilibrium (upright, stationary)
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])
    t_span = (0.0, duration)
    
    controller = create_controller(controller_type, poles=poles)
    
    # Print controller info
    info = controller.get_info()
    print(f"  Controller parameters:")
    for key, value in info.items():
        if key != 'type':
            print(f"    {key}: {value}")
    
    # Create observer if requested
    observer = None
    if use_observer:
        observer = LuenbergerObserver(observer_poles=observer_poles)
        obs_info = observer.get_info()
        print(f"  Observer parameters:")
        for key, value in obs_info.items():
            if key != 'type':
                print(f"    {key}: {value}")
    
    print("Running simulation with disturbance...")
    results = simulate(
        initial_state, t_span,
        controller=controller,
        enable_air_drag=enable_air_drag,
        observer=observer,
        noise_std_x=noise_std_x,
        noise_std_theta=noise_std_theta,
        disturbances=[disturbance]
    )
    
    t = results['t']
    states = results['states']
    control = results['control']
    
    # Pad control
    if len(control) < len(states):
        control = np.concatenate([[0.0], control])
    
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
    
    # Check if pendulum fell (θ > 90°)
    fell = np.any(theta_after > 90.0)
    
    # Max control force
    max_force = np.max(np.abs(control))
    
    if use_observer and 'estimates' in results:
        estimates = results['estimates']
        if len(estimates) < len(states):
            estimates = np.vstack([initial_state, estimates])
        
        estimation_error = states - estimates
        rms_error_x = np.sqrt(np.mean(estimation_error[:, 0]**2))
        rms_error_theta = np.sqrt(np.mean(estimation_error[:, 2]**2))
        print(f"\nEstimation error (RMS):")
        print(f"  Position: {rms_error_x*1000:.2f} mm")
        print(f"  Angle: {np.degrees(rms_error_theta):.2f}°")
    
    print(f"\nResults:")
    print(f"  Disturbance applied at: t={disturbance_time:.2f}s")
    print(f"  Max angle deviation: {max_deviation:.2f}°")
    if fell:
        print(f"  Status: FAILED - Pendulum fell over")
    elif settling_time:
        print(f"  Settling time: {settling_time:.2f}s")
        print(f"  Status: SUCCESS")
    else:
        print(f"  Settling time: Did not settle within {duration}s")
        print(f"  Status: PARTIAL - Stabilized but slow")
    print(f"  Max control force: {max_force:.2f} N")
    
    if save:
        data_dir = Path(__file__).parent / "data"
        drag_str = "drag" if enable_air_drag else "nodrag"
        obs_str = "obs" if use_observer else "perfect"
        filename_stem = f"eval_a{test_number}_{controller_type}_{drag_str}_{obs_str}"
        
        metadata = {
            'Evaluation': f'A{test_number} ({test_name})',
            'Controller': controller_type,
            'Air drag': enable_air_drag,
            'Observer': use_observer,
            'Disturbance time [s]': disturbance_time,
            'Max deviation [deg]': max_deviation,
            'Settling time [s]': settling_time if settling_time else 'N/A',
            'Fell': fell,
            'Max control force [N]': max_force,
        }
        metadata.update({f'Controller {k}': v for k, v in info.items()})
        if use_observer:
            metadata.update({f'Observer {k}': v for k, v in obs_info.items()})
            metadata['Noise std x [m]'] = noise_std_x
            metadata['Noise std theta [deg]'] = noise_std_theta
        
        save_results(data_dir / f"{filename_stem}.txt", t, states, get_parameters(), metadata)
        print(f"  Saved data to: {data_dir / filename_stem}.txt")
        
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        title_suffix = f" - A{test_number}: {test_name} ({controller_type.upper()})"
        
        fig1 = plot_time_series(t, states, control=control, disturbance=disturbance, title_suffix=title_suffix)
        fig1.savefig(plots_dir / f"{filename_stem}_timeseries.png", dpi=150, bbox_inches='tight')
        print(f"  Saved time series plot to: {plots_dir / filename_stem}_timeseries.png")
        
        fig2 = plot_phase_portrait(states, title_suffix=title_suffix)
        fig2.savefig(plots_dir / f"{filename_stem}_phase.png", dpi=150, bbox_inches='tight')
        print(f"  Saved phase portrait to: {plots_dir / filename_stem}_phase.png")
        
        plt.close('all')
    
    if show_animation:
        title = f"Eval A{test_number}: {controller_type.upper()} - {test_name}"
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
            disturbances=[disturbance],
            interval=20, skip_frames=2
        )
    
    return results


def run_eval_b(angle_deg, controller_type, duration=DEFAULT_DURATION, enable_air_drag=True,
               show_animation=True, show_sliders=False, save=True, poles=None,
               use_observer=False, observer_poles=None, 
               noise_std_x=DEFAULT_NOISE_STD_X,
               noise_std_theta=DEFAULT_NOISE_STD_THETA):
    print(f"Evaluation B: Recovery from {angle_deg}°")
    print(f"  Controller: {controller_type.upper()}")
    print(f"  Air drag: {'enabled' if enable_air_drag else 'disabled'}")
    print(f"  Observer: {'enabled' if use_observer else 'disabled'}")
    if use_observer:
        print(f"  Noise: σ_x={noise_std_x:.4f}m, σ_θ={noise_std_theta:.2f}°")
    print("-" * 50)

    theta_0 = np.radians(angle_deg)
    initial_state = np.array([0.0, 0.0, theta_0, 0.0])
    t_span = (0.0, duration)

    controller = create_controller(controller_type, poles=poles)
    info = controller.get_info()
    print(f"  Controller parameters:")
    for key, value in info.items():
        if key != 'type':
            print(f"    {key}: {value}")
    
    observer = None
    if use_observer:
        observer = LuenbergerObserver(observer_poles=observer_poles)
        obs_info = observer.get_info()
        print(f"  Observer parameters:")
        for key, value in obs_info.items():
            if key != 'type':
                print(f"    {key}: {value}")
    
    print("Running simulation...")
    results = simulate(initial_state, t_span, controller=controller, 
                       enable_air_drag=enable_air_drag, observer=observer,
                       noise_std_x=noise_std_x, noise_std_theta=noise_std_theta)
    
    t = results['t']
    states = results['states']
    control = results['control']
    
    # Pad control to match states (control computed after first state)
    if len(control) < len(states):
        control = np.concatenate([[0.0], control])
    
    final_theta_deg = np.abs(np.degrees(states[-100:, 2]))
    stabilised = np.all(final_theta_deg < 1.0)
    
    theta_deg = np.abs(np.degrees(states[:, 2]))
    settling_idx = None
    for i in range(len(theta_deg) - 50):
        if np.all(theta_deg[i:i+50] < 0.5):
            settling_idx = i
            break
    settling_time = t[settling_idx] if settling_idx else None
    
    max_force = np.max(np.abs(control[:len(t)]))
    
    if use_observer and 'estimates' in results:
        estimates = results['estimates']
        if len(estimates) < len(states):
            estimates = np.vstack([initial_state, estimates])
        
        estimation_error = states - estimates
        rms_error_x = np.sqrt(np.mean(estimation_error[:, 0]**2))
        rms_error_theta = np.sqrt(np.mean(estimation_error[:, 2]**2))
        print(f"\nEstimation error (RMS):")
        print(f"  Position: {rms_error_x*1000:.2f} mm")
        print(f"  Angle: {np.degrees(rms_error_theta):.2f}°")
    
    print(f"\nResults:")
    print(f"  Initial angle: {angle_deg}°")
    print(f"  Stabilised: {'Yes' if stabilised else 'No'}")
    if settling_time:
        print(f"  Settling time: {settling_time:.2f} s")
    else:
        print(f"  Settling time: Did not settle")
    print(f"  Final angle: {np.degrees(states[-1, 2]):.2f}°")
    print(f"  Final cart position: {states[-1, 0]:.3f} m")
    print(f"  Max control force: {max_force:.2f} N")
    
    if save:
        data_dir = Path(__file__).parent / "data"
        drag_str = "drag" if enable_air_drag else "nodrag"
        obs_str = "obs" if use_observer else "perfect"
        filename_stem = f"eval_b_{controller_type}_{drag_str}_{obs_str}_{abs(angle_deg)}deg"
        
        metadata = {
            'Evaluation': 'B (Recovery)',
            'Controller': controller_type,
            'Air drag': enable_air_drag,
            'Observer': use_observer,
            'Initial angle [deg]': angle_deg,
            'Stabilised': stabilised,
            'Settling time [s]': settling_time if settling_time else 'N/A',
            'Max control force [N]': max_force,
        }
        metadata.update({f'Controller {k}': v for k, v in info.items()})
        
        if use_observer:
            metadata.update({f'Observer {k}': v for k, v in obs_info.items()})
            metadata['Noise std x [m]'] = noise_std_x
            metadata['Noise std theta [deg]'] = noise_std_theta
        
        save_results(data_dir / f"{filename_stem}.txt", t, states, get_parameters(), metadata)
        print(f"  Saved data to: {data_dir / filename_stem}.txt")
        
        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        title_suffix = f" - B: Recovery from {angle_deg}° ({controller_type.upper()})"
        
        fig1 = plot_time_series(t, states, control=control, title_suffix=title_suffix)
        fig1.savefig(plots_dir / f"{filename_stem}_timeseries.png", dpi=150, bbox_inches='tight')
        print(f"  Saved time series plot to: {plots_dir / filename_stem}_timeseries.png")
        
        fig2 = plot_phase_portrait(states, title_suffix=title_suffix)
        fig2.savefig(plots_dir / f"{filename_stem}_phase.png", dpi=150, bbox_inches='tight')
        print(f"  Saved phase portrait to: {plots_dir / filename_stem}_phase.png")
        
        plt.close('all')
    
    if show_animation:
        title = f"Eval B: {controller_type.upper()} from {angle_deg}°"
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
            disturbances=None,
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
    
    # Set duration based on evaluation type
    if args.duration is not None:
        duration = args.duration
    elif args.A1 or args.A2 or args.A3:
        duration = EVAL_A_DURATION
    else:
        duration = DEFAULT_DURATION

    if args.A1:
        run_eval_a(
            test_number=1,
            controller_type=args.controller,
            duration=duration,
            enable_air_drag=not args.no_drag,
            show_animation=not args.no_animation,
            show_sliders=args.textboxes,
            save=not args.no_save,
            poles=poles,
            use_observer=args.observer,
            observer_poles=observer_poles,
            noise_std_x=args.noise_x,
            noise_std_theta=args.noise_theta,
        )
    elif args.A2:
        run_eval_a(
            test_number=2,
            controller_type=args.controller,
            duration=duration,
            enable_air_drag=not args.no_drag,
            show_animation=not args.no_animation,
            show_sliders=args.textboxes,
            save=not args.no_save,
            poles=poles,
            use_observer=args.observer,
            observer_poles=observer_poles,
            noise_std_x=args.noise_x,
            noise_std_theta=args.noise_theta,
        )
    elif args.A3:
        run_eval_a(
            test_number=3,
            controller_type=args.controller,
            duration=duration,
            enable_air_drag=not args.no_drag,
            show_animation=not args.no_animation,
            show_sliders=args.textboxes,
            save=not args.no_save,
            poles=poles,
            use_observer=args.observer,
            observer_poles=observer_poles,
            noise_std_x=args.noise_x,
            noise_std_theta=args.noise_theta,
        )
    elif args.B is not None:
        run_eval_b(
            angle_deg=args.B,
            controller_type=args.controller,
            duration=duration,
            enable_air_drag=not args.no_drag,
            show_animation=not args.no_animation,
            show_sliders=args.textboxes,
            save=not args.no_save,
            poles=poles,
            use_observer=args.observer,
            observer_poles=observer_poles,
            noise_std_x=args.noise_x,
            noise_std_theta=args.noise_theta,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()