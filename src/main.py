import argparse
import numpy as np
from pathlib import Path

from dynamics import get_parameters
from controllers import LQRController
from simulation import simulate, save_results
from visualisation import animate_from_arrays


FORCE_LIMITS = (-50.0, 50.0)


def run_eval_b(angle_deg, duration=10.0, show_animation=True, save=True):
    print(f"Evaluation B: Recovery from {angle_deg}°")
    
    theta_0 = np.radians(angle_deg)
    initial_state = np.array([0.0, 0.0, theta_0, 0.0])
    t_span = (0.0, duration)
    
    controller = LQRController(output_limits=FORCE_LIMITS)
    
    print("Running simulation.")
    t, states, control = simulate(initial_state, t_span, controller=controller)
    
    # Check if stabilised (theta within 1 degree for last second)
    final_theta_deg = np.abs(np.degrees(states[-100:, 2]))
    stabilised = np.all(final_theta_deg < 1.0)
    
    # Find settling time (theta stays within 2 degrees for 0.5s)
    theta_deg = np.abs(np.degrees(states[:, 2]))
    settling_idx = None
    for i in range(len(theta_deg) - 50):
        if np.all(theta_deg[i:i+50] < 2.0):
            settling_idx = i
            break
    settling_time = t[settling_idx] if settling_idx else None
    
    print(f"  Initial angle: {angle_deg}°")
    print(f"  Stabilised: {'Yes' if stabilised else 'No'}")
    if settling_time:
        print(f"  Settling time: {settling_time:.2f} s")
    else:
        print(f"  Settling time: Did not settle")
    print(f"  Final angle: {np.degrees(states[-1, 2]):.2f}°")
    print(f"  Final cart position: {states[-1, 0]:.3f} m")
    
    if save:
        data_dir = Path(__file__).parent.parent / "data"
        filename = f"eval_b_{abs(angle_deg)}deg.txt"
        metadata = {
            'Evaluation': 'B (Recovery)',
            'Initial angle [deg]': angle_deg,
            'Stabilised': stabilised,
            'Settling time [s]': settling_time if settling_time else 'N/A',
        }
        save_results(data_dir / filename, t, states, get_parameters(), metadata)
        print(f"  Saved to: {data_dir / filename}")
    
    if show_animation:
        title = f"Eval B: Recovery from {angle_deg}°"
        animate_from_arrays(t, states, title=title, interval=20, skip_frames=2)
    
    return {
        'stabilised': stabilised,
        'settling_time': settling_time,
        'final_angle_deg': np.degrees(states[-1, 2]),
        'final_x': states[-1, 0],
        't': t,
        'states': states,
        'control': control,
    }


def main():
    parser = argparse.ArgumentParser(description='Cart-Pendulum Control Evaluation')
    parser.add_argument('-B', type=float, metavar='ANGLE',
                        help='Run evaluation B (recovery) from specified angle in degrees')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Simulation duration in seconds')
    parser.add_argument('--no-animation', action='store_true',
                        help='Disable animation')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable saving results')
    
    args = parser.parse_args()
    
    if args.B is not None:
        run_eval_b(
            angle_deg=args.B,
            duration=args.duration,
            show_animation=not args.no_animation,
            save=not args.no_save,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()