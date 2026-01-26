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


def create_controller(controller_type, poles=None):
    if controller_type == 'lqr':
        return LQRController(output_limits=FORCE_LIMITS)
    elif controller_type == 'pid':
        return PIDController(output_limits=FORCE_LIMITS)
    elif controller_type == 'pole':
        return PolePlacementController(poles=poles, output_limits=FORCE_LIMITS)


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
            interval=20, skip_frames=2
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Cart-Pendulum Control Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # main arguments
    parser.add_argument('-B', type=float, metavar='ANGLE')
    parser.add_argument('--controller', type=str, required=True, choices=['lqr', 'pid', 'pole'])
    parser.add_argument('--poles', type=str, default=None)
    parser.add_argument('--textboxes', action='store_false')

    # observer arguments
    parser.add_argument('--observer', action='store_false')
    parser.add_argument('--observer-poles', type=str, default=None)

    # noise arguments
    parser.add_argument('--noise-x', type=float, default=DEFAULT_NOISE_STD_X, help='Position noise [m]')
    parser.add_argument('--noise-theta', type=float, default=DEFAULT_NOISE_STD_THETA, help='Angle noise [deg]')

    # other arguments
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION)
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

    if args.B is not None:
        run_eval_b(
            angle_deg=args.B,
            controller_type=args.controller,
            duration=args.duration,
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