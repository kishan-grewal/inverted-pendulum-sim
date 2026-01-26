import numpy as np
import matplotlib.pyplot as plt


def plot_time_series(t, states, control=None, disturbance=None, title_suffix=""):
    """
    Plot time series of all states and optionally control force.
    """
    if control is not None:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    fig.suptitle(f"Cart-Pendulum Time Series{title_suffix}")
    
    axes_flat = axes.flatten()
    
    # Cart position
    axes_flat[0].plot(t, states[:, 0], 'b-', linewidth=1)
    axes_flat[0].set_xlabel('Time [s]')
    axes_flat[0].set_ylabel('Cart position x [m]')
    axes_flat[0].set_title('Cart Position')
    axes_flat[0].grid(True, alpha=0.3)
    
    # Cart velocity
    axes_flat[1].plot(t, states[:, 1], 'b-', linewidth=1)
    axes_flat[1].set_xlabel('Time [s]')
    axes_flat[1].set_ylabel('Cart velocity ẋ [m/s]')
    axes_flat[1].set_title('Cart Velocity')
    axes_flat[1].grid(True, alpha=0.3)
    
    # Pendulum angle
    theta_deg = np.degrees(states[:, 2])
    axes_flat[2].plot(t, theta_deg, 'r-', linewidth=1)
    axes_flat[2].set_xlabel('Time [s]')
    axes_flat[2].set_ylabel('Pendulum angle θ [°]')
    axes_flat[2].set_title('Pendulum Angle')
    axes_flat[2].axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Upright')
    axes_flat[2].grid(True, alpha=0.3)
    axes_flat[2].legend()
    
    # Pendulum angular velocity
    theta_dot_deg = np.degrees(states[:, 3])
    axes_flat[3].plot(t, theta_dot_deg, 'r-', linewidth=1)
    axes_flat[3].set_xlabel('Time [s]')
    axes_flat[3].set_ylabel('Angular velocity θ̇ [°/s]')
    axes_flat[3].set_title('Pendulum Angular Velocity')
    axes_flat[3].grid(True, alpha=0.3)
    
    # Control force (if provided)
    if control is not None:
        axes_flat[4].plot(t, control, 'g-', linewidth=1)
        axes_flat[4].set_xlabel('Time [s]')
        axes_flat[4].set_ylabel('Control force F [N]')
        axes_flat[4].set_title('Control Force')
        axes_flat[4].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_flat[4].grid(True, alpha=0.3)
        
        # Disturbance plot (if provided)
        if disturbance is not None:
            dist_time, cart_impulse, angular_impulse = disturbance
            
            if angular_impulse != 0:
                # A1/A2: Show angular disturbance on pendulum
                axes_flat[5].axvline(x=dist_time, color='red', linewidth=3, alpha=0.7,
                                    label=f'Impulse: {angular_impulse} N·s·m')
                axes_flat[5].set_ylabel('Angular impulse [N·s·m]')
                axes_flat[5].set_title('Pendulum Disturbance')
                axes_flat[5].set_ylim(-0.1, 0.3)
            else:
                # A3: Show cart disturbance
                axes_flat[5].axvline(x=dist_time, color='blue', linewidth=3, alpha=0.7,
                                    label=f'Impulse: {cart_impulse} N·s')
                axes_flat[5].set_ylabel('Cart impulse [N·s]')
                axes_flat[5].set_title('Cart Disturbance')
                axes_flat[5].set_ylim(-0.5, 4.0)
            
            axes_flat[5].set_xlabel('Time [s]')
            axes_flat[5].set_xlim(t[0], t[-1])
            axes_flat[5].grid(True, alpha=0.3)
            axes_flat[5].legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_phase_portrait(states, title_suffix=""):
    """Plot phase portraits for cart and pendulum."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Phase Portraits{title_suffix}")
    
    # Cart phase portrait
    axes[0].plot(states[:, 0], states[:, 1], 'b-', linewidth=0.5, alpha=0.7)
    axes[0].plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')
    axes[0].plot(states[-1, 0], states[-1, 1], 'ro', markersize=10, label='End')
    axes[0].set_xlabel('Cart position x [m]')
    axes[0].set_ylabel('Cart velocity ẋ [m/s]')
    axes[0].set_title('Cart Phase Portrait')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_aspect('auto')
    
    # Pendulum phase portrait (wrapped to [-pi, pi])
    theta_wrapped = np.arctan2(np.sin(states[:, 2]), np.cos(states[:, 2]))
    theta_dot = states[:, 3]
    
    axes[1].plot(np.degrees(theta_wrapped), np.degrees(theta_dot), 'r-', linewidth=0.5, alpha=0.7)
    axes[1].plot(np.degrees(theta_wrapped[0]), np.degrees(theta_dot[0]), 'go', markersize=10, label='Start')
    axes[1].plot(np.degrees(theta_wrapped[-1]), np.degrees(theta_dot[-1]), 'ro', markersize=10, label='End')
    axes[1].set_xlabel('Pendulum angle θ [°]')
    axes[1].set_ylabel('Angular velocity θ̇ [°/s]')
    axes[1].set_title('Pendulum Phase Portrait (wrapped to ±180°)')
    axes[1].axvline(x=0, color='g', linestyle='--', alpha=0.3, label='Upright')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    return fig