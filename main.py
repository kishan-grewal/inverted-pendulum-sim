import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.dynamics import get_parameters
from src.controllers import LQRController, PIDController
from src.simulation import simulate, save_results
from src.visualisation import animate_from_arrays


FORCE_LIMITS = (-15.0, 15.0)


def create_controller(controller_type):
    """Create controller instance based on type string."""
    if controller_type == 'lqr':
        return LQRController(output_limits=FORCE_LIMITS)
    elif controller_type == 'pid':
        return PIDController(output_limits=FORCE_LIMITS)
    else:
        raise ValueError(f"Unknown controller: {controller_type}")


def run_eval_b(angle_deg, controller_type, duration=10.0, enable_air_drag=True,
               show_animation=True, show_sliders=False, save=True):
    """
    Run Evaluation B: Recovery from initial angle offset.
    
    Args:
        angle_deg: initial angle in degrees
        controller_type: 'lqr' or 'pid'
        duration: simulation duration [s]
        enable_air_drag: include air drag in simulation
        show_animation: display animation window
        show_sliders: show PID gain sliders (only for PID controller)
        save: save results to file
    """
    print(f"Evaluation B: Recovery from {angle_deg}°")
    print(f"  Controller: {controller_type.upper()}")
    print(f"  Air drag: {'enabled' if enable_air_drag else 'disabled'}")
    print("-" * 50)
    
    theta_0 = np.radians(angle_deg)
    initial_state = np.array([0.0, 0.0, theta_0, 0.0])
    t_span = (0.0, duration)
    
    controller = create_controller(controller_type)
    
    # Print controller info
    info = controller.get_info()
    print(f"  Controller parameters:")
    for key, value in info.items():
        if key != 'type':
            print(f"    {key}: {value}")
    
    print("Running simulation...")
    results = simulate(initial_state, t_span, controller=controller, 
                       enable_air_drag=enable_air_drag)
    
    t = results['t']
    states = results['states']
    
    # Check if stabilised (final 1 second within ±1°)
    final_theta_deg = np.abs(np.degrees(states[-100:, 2]))
    stabilised = np.all(final_theta_deg < 1.0)
    
    # Find settling time (when angle stays within ±2° for 0.5s)
    theta_deg = np.abs(np.degrees(states[:, 2]))
    settling_idx = None
    for i in range(len(theta_deg) - 50):
        if np.all(theta_deg[i:i+50] < 2.0):
            settling_idx = i
            break
    settling_time = t[settling_idx] if settling_idx else None
    
    # Compute max control force used
    max_force = np.max(np.abs(results['control'][:len(t)]))
    
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
        filename = f"eval_b_{controller_type}_{drag_str}_{abs(angle_deg)}deg.txt"
        metadata = {
            'Evaluation': 'B (Recovery)',
            'Controller': controller_type,
            'Air drag': enable_air_drag,
            'Initial angle [deg]': angle_deg,
            'Stabilised': stabilised,
            'Settling time [s]': settling_time if settling_time else 'N/A',
            'Max control force [N]': max_force,
        }
        # Add controller-specific info
        metadata.update({f'Controller {k}': v for k, v in info.items()})
        
        save_results(data_dir / filename, t, states, get_parameters(), metadata)
        print(f"  Saved to: {data_dir / filename}")
    
    if show_animation:
        title = f"Eval B: {controller_type.upper()} from {angle_deg}°"
        if not enable_air_drag:
            title += " (no drag)"
        
        # Only show sliders for PID
        use_sliders = show_sliders and controller_type == 'pid'
        animate_from_arrays(t, states, title=title, 
                           controller=controller if use_sliders else None,
                           show_sliders=use_sliders,
                           interval=20, skip_frames=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Cart-Pendulum Control Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -B 15 --controller lqr
  python main.py -B 15 --controller pid --sliders
  python main.py -B 15 --controller pid --no-drag
        """
    )
    parser.add_argument('-B', type=float, metavar='ANGLE',
                        help='Run evaluation B (recovery) from specified angle in degrees')
    parser.add_argument('--controller', type=str, required=True, 
                        choices=['lqr', 'pid'],
                        help='Controller type: lqr or pid')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Simulation duration in seconds (default: 10)')
    parser.add_argument('--no-drag', action='store_true',
                        help='Disable air drag simulation')
    parser.add_argument('--sliders', action='store_true',
                        help='Show PID gain sliders (only for PID controller)')
    parser.add_argument('--no-animation', action='store_true',
                        help='Disable animation')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable saving results')
    
    args = parser.parse_args()
    
    if args.B is not None:
        run_eval_b(
            angle_deg=args.B,
            controller_type=args.controller,
            duration=args.duration,
            enable_air_drag=not args.no_drag,
            show_animation=not args.no_animation,
            show_sliders=args.sliders,
            save=not args.no_save,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()