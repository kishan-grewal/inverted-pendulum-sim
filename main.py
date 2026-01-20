import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.dynamics import get_parameters
from src.controllers import LQRController
from src.observers import LuenbergerObserver
from src.simulation import simulate, save_results
from src.visualisation import animate_from_arrays


FORCE_LIMITS = (-1000.0, 1000.0)


def create_controller(controller_type):
    if controller_type == 'lqr':
        return LQRController(output_limits=FORCE_LIMITS)
    else:
        raise ValueError(f"Unknown controller: {controller_type}")


def create_observer(observer_type):
    if observer_type == 'luenberger':
        return LuenbergerObserver()
    else:
        raise ValueError(f"Unknown observer: {observer_type}")


def plot_estimation_comparison(t, true_states, estimated_states, title_suffix=""):
    n = min(len(true_states), len(estimated_states))
    t_plot = t[:n]
    true_plot = true_states[:n]
    est_plot = estimated_states[:n]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"State Estimation Comparison{title_suffix}")
    
    labels = ['x [m]', 'ẋ [m/s]', 'θ [rad]', 'θ̇ [rad/s]']
    
    for i, ax in enumerate(axes.flat):
        ax.plot(t_plot, true_plot[:, i], 'b-', label='True', linewidth=1.5)
        ax.plot(t_plot, est_plot[:, i], 'r--', label='Estimated', linewidth=1.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(labels[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def run_eval_b(angle_deg, controller_type, observer_type, duration=10.0, show_animation=True, save=True):
    print(f"Evaluation B: Recovery from {angle_deg}°")
    print(f"  Controller: {controller_type}")
    print(f"  Observer: {observer_type}")
    print("-" * 50)
    
    theta_0 = np.radians(angle_deg)
    initial_state = np.array([0.0, 0.0, theta_0, 0.0])
    t_span = (0.0, duration)
    
    controller = create_controller(controller_type)
    observer = create_observer(observer_type)
    
    print("Running simulation...")
    results = simulate(initial_state, t_span, controller=controller, observer=observer)
    
    t = results['t']
    states = results['states']
    estimated_states = results['estimated_states']
    
    # Check if stabilised
    final_theta_deg = np.abs(np.degrees(states[-100:, 2]))
    stabilised = np.all(final_theta_deg < 1.0)
    
    # Find settling time
    theta_deg = np.abs(np.degrees(states[:, 2]))
    settling_idx = None
    for i in range(len(theta_deg) - 50):
        if np.all(theta_deg[i:i+50] < 2.0):
            settling_idx = i
            break
    settling_time = t[settling_idx] if settling_idx else None
    
    # Compute estimation RMSE
    n = min(len(states), len(estimated_states))
    rmse = np.sqrt(np.mean((states[:n] - estimated_states[:n]) ** 2, axis=0))
    
    print(f"  Initial angle: {angle_deg}°")
    print(f"  Stabilised: {'Yes' if stabilised else 'No'}")
    if settling_time:
        print(f"  Settling time: {settling_time:.2f} s")
    else:
        print(f"  Settling time: Did not settle")
    print(f"  Final angle: {np.degrees(states[-1, 2]):.2f}°")
    print(f"  Final cart position: {states[-1, 0]:.3f} m")
    print(f"  Estimation RMSE: x={rmse[0]:.4f}, x_dot={rmse[1]:.4f}, theta={rmse[2]:.4f}, theta_dot={rmse[3]:.4f}")
    
    if save:
        data_dir = Path(__file__).parent.parent / "data"
        filename = f"eval_b_{controller_type}_{observer_type}_{abs(angle_deg)}deg.txt"
        metadata = {
            'Evaluation': 'B (Recovery)',
            'Controller': controller_type,
            'Observer': observer_type,
            'Initial angle [deg]': angle_deg,
            'Stabilised': stabilised,
            'Settling time [s]': settling_time if settling_time else 'N/A',
            'RMSE x': rmse[0],
            'RMSE x_dot': rmse[1],
            'RMSE theta': rmse[2],
            'RMSE theta_dot': rmse[3],
        }
        save_results(data_dir / filename, t, states, get_parameters(), metadata)
        print(f"  Saved to: {data_dir / filename}")
        
        # Save comparison plot
        plots_dir = Path(__file__).parent.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_estimation_comparison(t, states, estimated_states, f" (θ₀={angle_deg}°)")
        plot_filename = f"eval_b_{controller_type}_{observer_type}_{abs(angle_deg)}deg_estimation.png"
        fig.savefig(plots_dir / plot_filename, dpi=150)
        print(f"  Saved plot to: {plots_dir / plot_filename}")
        plt.close(fig)
    
    if show_animation:
        title = f"Eval B: {controller_type.upper()} + {observer_type} from {angle_deg}°"
        animate_from_arrays(t, states, title=title, interval=20, skip_frames=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Cart-Pendulum Control Evaluation')
    parser.add_argument('-B', type=float, metavar='ANGLE',
                        help='Run evaluation B (recovery) from specified angle in degrees')
    parser.add_argument('--controller', type=str, required=True, choices=['lqr'],
                        help='Controller type')
    parser.add_argument('--observer', type=str, required=True, choices=['luenberger'],
                        help='Observer type')
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
            controller_type=args.controller,
            observer_type=args.observer,
            duration=args.duration,
            show_animation=not args.no_animation,
            save=not args.no_save,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()