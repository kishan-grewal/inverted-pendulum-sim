import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.widgets import Slider, Button
from pathlib import Path

from src.dynamics import get_parameters, L_rod


# -----------------------------------------------------------------------------
# Static plotting from saved data
# -----------------------------------------------------------------------------

def load_results(filepath):
    """Load simulation results from text file."""
    filepath = Path(filepath)
    
    t_list = []
    states_list = []
    metadata = {'parameters': {}}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# Initial angle:'):
                parts = line.split(':')[1].strip().split()
                metadata['initial_angle_deg'] = float(parts[0])
            elif line.startswith('#   '):
                param_part = line[4:]
                if ':' in param_part:
                    key, value = param_part.split(':')
                    try:
                        metadata['parameters'][key.strip()] = float(value.strip())
                    except ValueError:
                        pass
            elif line and not line.startswith('#'):
                values = [float(v) for v in line.split()]
                t_list.append(values[0])
                states_list.append(values[1:])
    
    return np.array(t_list), np.array(states_list), metadata


def plot_time_series(t, states, title_suffix=""):
    """Plot time series of all states."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Cart-Pendulum Time Series{title_suffix}")
    
    # Cart position
    axes[0, 0].plot(t, states[:, 0], 'b-', linewidth=1)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Cart position x [m]')
    axes[0, 0].set_title('Cart Position')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cart velocity
    axes[0, 1].plot(t, states[:, 1], 'b-', linewidth=1)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Cart velocity ẋ [m/s]')
    axes[0, 1].set_title('Cart Velocity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pendulum angle
    theta_deg = np.degrees(states[:, 2])
    axes[1, 0].plot(t, theta_deg, 'r-', linewidth=1)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Pendulum angle θ [°]')
    axes[1, 0].set_title('Pendulum Angle')
    axes[1, 0].axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Upright')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Pendulum angular velocity
    theta_dot_deg = np.degrees(states[:, 3])
    axes[1, 1].plot(t, theta_dot_deg, 'r-', linewidth=1)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Angular velocity θ̇ [°/s]')
    axes[1, 1].set_title('Pendulum Angular Velocity')
    axes[1, 1].grid(True, alpha=0.3)
    
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


def plot_from_file(filepath, show=True, save=False):
    """Load and plot results from file."""
    t, states, metadata = load_results(filepath)
    
    angle = metadata.get('initial_angle_deg', '?')
    title_suffix = f" (θ₀ = {angle}°)"
    
    fig1 = plot_time_series(t, states, title_suffix)
    fig2 = plot_phase_portrait(states, title_suffix)
    
    if save:
        plots_dir = Path(filepath).parent.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(filepath).stem
        fig1.savefig(plots_dir / f"{stem}_time_series.png", dpi=150)
        fig2.savefig(plots_dir / f"{stem}_phase_portrait.png", dpi=150)
        print(f"Saved plots to {plots_dir}")
    
    if show:
        plt.show()
    
    return fig1, fig2


# -----------------------------------------------------------------------------
# Real-time animation
# -----------------------------------------------------------------------------

class CartPendulumAnimator:
    """Animator for cart-pendulum system with optional controller and noise sliders."""

    def __init__(self, t, states, cart_width=0.3, cart_height=0.15,
                 pendulum_length=None, title="Cart-Pendulum Animation",
                 controller=None, observer=None, show_sliders=False,
                 initial_state=None, t_span=None, enable_air_drag=True,
                 control=None, noise_std_x=0.002, noise_std_theta=0.005):
        self.t = t
        self.states = states
        self.control = control if control is not None else np.zeros(len(t))
        self.cart_width = cart_width
        self.cart_height = cart_height
        self.pendulum_length = pendulum_length if pendulum_length else L_rod
        self.title = title
        self.controller = controller
        self.observer = observer
        self.show_sliders = show_sliders and (controller is not None or observer is not None)

        # Store simulation parameters for re-running
        self.initial_state = initial_state
        self.t_span = t_span
        self.enable_air_drag = enable_air_drag
        self.noise_std_x = noise_std_x
        self.noise_std_theta = noise_std_theta

        # Determine controller type
        self.controller_type = None
        if controller is not None:
            info = controller.get_info()
            if 'PID' in info['type']:
                self.controller_type = 'pid'
            elif 'LQR' in info['type']:
                self.controller_type = 'lqr'
            elif 'PolePlacement' in info['type'] or 'Pole' in info['type']:
                self.controller_type = 'pole'

        # Current frame index
        self.current_frame = 0
        self.skip_frames = 1

        self.fig = None
        self.ax_anim = None
        self.ax_theta = None
        self.ax_x = None
        self.ax_force = None
        self.ax_dforce = None
        self.anim = None

        # Slider references
        self.sliders = {}
    
    def _compute_axis_limits(self):
        """Compute axis limits from current data."""
        x_min = np.min(self.states[:, 0]) - self.cart_width - self.pendulum_length
        x_max = np.max(self.states[:, 0]) + self.cart_width + self.pendulum_length
        y_min = -self.pendulum_length - self.cart_height
        y_max = self.pendulum_length + self.cart_height
        
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        self.xlim = (x_min - x_padding, x_max + x_padding)
        self.ylim = (y_min - y_padding, y_max + y_padding)
    
    def _setup_figure(self):
        """Create figure with animation and plots."""
        self._compute_axis_limits()

        if self.show_sliders:
            self.fig = plt.figure(figsize=(20, 10))
        else:
            self.fig = plt.figure(figsize=(16, 9))
        self.fig.suptitle(self.title)

        # Adjust layout for sliders if needed
        if self.show_sliders:
            # Main animation axes
            self.ax_anim = self.fig.add_axes([0.03, 0.30, 0.38, 0.60])

            # Right side: 2x2 grid of plots
            self.ax_theta = self.fig.add_axes([0.46, 0.55, 0.24, 0.35])
            self.ax_force = self.fig.add_axes([0.74, 0.55, 0.24, 0.35])
            self.ax_x = self.fig.add_axes([0.46, 0.10, 0.24, 0.35])
            self.ax_dforce = self.fig.add_axes([0.74, 0.10, 0.24, 0.35])

            # Create sliders based on controller type
            self._setup_sliders()

        else:
            # Standard layout without sliders
            self.ax_anim = self.fig.add_axes([0.03, 0.10, 0.40, 0.80])
            self.ax_theta = self.fig.add_axes([0.50, 0.55, 0.22, 0.35])
            self.ax_force = self.fig.add_axes([0.76, 0.55, 0.22, 0.35])
            self.ax_x = self.fig.add_axes([0.50, 0.10, 0.22, 0.35])
            self.ax_dforce = self.fig.add_axes([0.76, 0.10, 0.22, 0.35])

        self._setup_axes()
        self._setup_artists()
    
    def _setup_sliders(self):
        """Create sliders based on controller type and observer."""
        slider_y_start = 0.22
        slider_height = 0.02
        slider_spacing = 0.03
        
        current_y = slider_y_start
        
        # Controller-specific sliders
        if self.controller_type == 'pid':
            info = self.controller.get_info()
            
            # PID sliders
            ax_kp = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['Kp'] = Slider(ax_kp, 'Kp', 0.0, 200.0, 
                                        valinit=info.get('Kp', 50.0), valstep=1.0)
            current_y -= slider_spacing
            
            ax_ki = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['Ki'] = Slider(ax_ki, 'Ki', 0.0, 10.0, 
                                        valinit=info.get('Ki', 0.0), valstep=0.1)
            current_y -= slider_spacing
            
            ax_kd = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['Kd'] = Slider(ax_kd, 'Kd', 0.0, 50.0, 
                                        valinit=info.get('Kd', 3.0), valstep=0.5)
            current_y -= slider_spacing
        
        elif self.controller_type == 'lqr':
            info = self.controller.get_info()
            weights = info.get('tuning_weights', {})
            
            # LQR weight sliders
            ax_pos = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['position_weight'] = Slider(ax_pos, 'Position Weight', 0.1, 10.0,
                                                      valinit=weights.get('position_weight', 1.0),
                                                      valstep=0.1)
            current_y -= slider_spacing
            
            ax_vel = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['velocity_weight'] = Slider(ax_vel, 'Velocity Weight', 0.1, 10.0,
                                                      valinit=weights.get('velocity_weight', 1.0),
                                                      valstep=0.1)
            current_y -= slider_spacing
            
            ax_ang = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['angle_weight'] = Slider(ax_ang, 'Angle Weight', 0.1, 10.0,
                                                   valinit=weights.get('angle_weight', 1.0),
                                                   valstep=0.1)
            current_y -= slider_spacing
            
            ax_angvel = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['angular_velocity_weight'] = Slider(ax_angvel, 'Angular Vel Weight', 0.1, 10.0,
                                                              valinit=weights.get('angular_velocity_weight', 1.0),
                                                              valstep=0.1)
            current_y -= slider_spacing
            
            ax_ctrl = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['control_weight'] = Slider(ax_ctrl, 'Control Weight', 0.1, 10.0,
                                                     valinit=weights.get('control_weight', 1.0),
                                                     valstep=0.1)
            current_y -= slider_spacing
        
        elif self.controller_type == 'pole':
            info = self.controller.get_info()
            poles = info.get('desired_poles', [-2, -3, -4, -5])
            
            # Pole placement sliders (4 real poles)
            ax_p1 = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['pole1'] = Slider(ax_p1, 'Pole 1', -20.0, -0.5,
                                           valinit=poles[0], valstep=0.5)
            current_y -= slider_spacing
            
            ax_p2 = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['pole2'] = Slider(ax_p2, 'Pole 2', -20.0, -0.5,
                                           valinit=poles[1], valstep=0.5)
            current_y -= slider_spacing
            
            ax_p3 = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['pole3'] = Slider(ax_p3, 'Pole 3', -20.0, -0.5,
                                           valinit=poles[2], valstep=0.5)
            current_y -= slider_spacing
            
            ax_p4 = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['pole4'] = Slider(ax_p4, 'Pole 4', -20.0, -0.5,
                                           valinit=poles[3], valstep=0.5)
            current_y -= slider_spacing
        
        # Noise sliders (universal if observer is present)
        if self.observer is not None:
            current_y -= slider_spacing * 0.5  # Extra gap before noise sliders
            
            ax_noise_x = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['noise_x'] = Slider(ax_noise_x, 'Noise σ_x [mm]', 0.0, 10.0,
                                             valinit=self.noise_std_x * 1000, valstep=0.1)
            current_y -= slider_spacing
            
            ax_noise_theta = self.fig.add_axes([0.10, current_y, 0.25, slider_height])
            self.sliders['noise_theta'] = Slider(ax_noise_theta, 'Noise σ_θ [°]', 0.0, 2.0,
                                                  valinit=np.degrees(self.noise_std_theta), valstep=0.05)
        
        # Connect all slider callbacks
        for slider in self.sliders.values():
            slider.on_changed(self._on_slider_change)
        
        # Add note about sliders
        self.fig.text(0.03, 0.01,
                     "Adjust sliders to re-run simulation with new parameters",
                     fontsize=9, style='italic', alpha=0.7)
    
    def _compute_dforce(self):
        """Compute derivative of force (dF/dt)."""
        dt = np.diff(self.t)
        dt = np.where(dt == 0, 1e-6, dt)
        dforce = np.diff(self.control[:len(self.t)]) / dt
        # Pad to match length of t
        return np.concatenate([[0], dforce])

    def _setup_axes(self):
        """Configure all axes."""
        # Configure animation axes
        self.ax_anim.set_xlim(self.xlim)
        self.ax_anim.set_ylim(self.ylim)
        self.ax_anim.set_aspect('equal')
        self.ax_anim.set_xlabel('x [m]')
        self.ax_anim.set_ylabel('y [m]')
        self.ax_anim.grid(True, alpha=0.3)
        self.ax_anim.axhline(y=-self.cart_height / 2, color='brown', linewidth=2)

        # Configure theta plot
        self.ax_theta.set_xlim(0, self.t[-1])
        theta_deg = np.degrees(self.states[:, 2])
        self.ax_theta.set_ylim(min(theta_deg.min(), -10) - 5, max(theta_deg.max(), 10) + 5)
        self.ax_theta.set_xlabel('Time [s]')
        self.ax_theta.set_ylabel('θ [°]')
        self.ax_theta.set_title('Pendulum Angle')
        self.ax_theta.grid(True, alpha=0.3)
        self.ax_theta.axhline(y=0, color='g', linestyle='--', alpha=0.5)

        # Configure x position plot
        self.ax_x.set_xlim(0, self.t[-1])
        self.ax_x.set_ylim(self.states[:, 0].min() - 0.1, self.states[:, 0].max() + 0.1)
        self.ax_x.set_xlabel('Time [s]')
        self.ax_x.set_ylabel('x [m]')
        self.ax_x.set_title('Cart Position')
        self.ax_x.grid(True, alpha=0.3)

        # Configure force plot
        force_data = self.control[:len(self.t)]
        self.ax_force.set_xlim(0, self.t[-1])
        force_margin = max(abs(force_data.min()), abs(force_data.max()), 1) * 0.1
        self.ax_force.set_ylim(force_data.min() - force_margin, force_data.max() + force_margin)
        self.ax_force.set_xlabel('Time [s]')
        self.ax_force.set_ylabel('F [N]')
        self.ax_force.set_title('Control Force')
        self.ax_force.grid(True, alpha=0.3)
        self.ax_force.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Configure dF/dt plot
        self.dforce = self._compute_dforce()
        self.ax_dforce.set_xlim(0, self.t[-1])
        dforce_margin = max(abs(self.dforce.min()), abs(self.dforce.max()), 1) * 0.1
        self.ax_dforce.set_ylim(self.dforce.min() - dforce_margin, self.dforce.max() + dforce_margin)
        self.ax_dforce.set_xlabel('Time [s]')
        self.ax_dforce.set_ylabel('dF/dt [N/s]')
        self.ax_dforce.set_title('Force Rate of Change')
        self.ax_dforce.grid(True, alpha=0.3)
        self.ax_dforce.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    def _setup_artists(self):
        """Create all plot artists."""
        # Cart
        self.cart_patch = FancyBboxPatch(
            (0, 0), self.cart_width, self.cart_height,
            boxstyle="round,pad=0.01",
            facecolor='steelblue',
            edgecolor='black',
            linewidth=2,
        )
        self.ax_anim.add_patch(self.cart_patch)

        # Wheels
        wheel_radius = self.cart_height * 0.25
        self.wheel_left = Circle((0, 0), wheel_radius, facecolor='black')
        self.wheel_right = Circle((0, 0), wheel_radius, facecolor='black')
        self.ax_anim.add_patch(self.wheel_left)
        self.ax_anim.add_patch(self.wheel_right)

        # Pendulum rod
        self.pendulum_line, = self.ax_anim.plot([], [], 'o-', color='firebrick',
                                                  linewidth=4, markersize=12,
                                                  markerfacecolor='darkred')

        # Pivot point
        self.pivot_point = Circle((0, 0), 0.02, facecolor='black', zorder=5)
        self.ax_anim.add_patch(self.pivot_point)

        # Time text
        self.time_text = self.ax_anim.text(0.02, 0.98, '', transform=self.ax_anim.transAxes,
                                            fontsize=12, verticalalignment='top',
                                            fontfamily='monospace')

        # Live plot lines
        self.theta_line, = self.ax_theta.plot([], [], 'r-', linewidth=1.5)
        self.x_line, = self.ax_x.plot([], [], 'b-', linewidth=1.5)
        self.force_line, = self.ax_force.plot([], [], 'g-', linewidth=1.5)
        self.dforce_line, = self.ax_dforce.plot([], [], 'm-', linewidth=1.5)

        # Current position markers
        self.theta_marker, = self.ax_theta.plot([], [], 'ro', markersize=8)
        self.x_marker, = self.ax_x.plot([], [], 'bo', markersize=8)
        self.force_marker, = self.ax_force.plot([], [], 'go', markersize=8)
        self.dforce_marker, = self.ax_dforce.plot([], [], 'mo', markersize=8)
    
    def _on_slider_change(self, val):
        """Callback for slider changes - re-run simulation and restart animation."""
        if self.controller is None and self.observer is None:
            return

        # Update controller parameters
        if self.controller is not None:
            if self.controller_type == 'pid':
                self.controller.set_gains(
                    Kp=self.sliders['Kp'].val,
                    Ki=self.sliders['Ki'].val,
                    Kd=self.sliders['Kd'].val
                )
            elif self.controller_type == 'lqr':
                self.controller.set_weights(
                    position_weight=self.sliders['position_weight'].val,
                    velocity_weight=self.sliders['velocity_weight'].val,
                    angle_weight=self.sliders['angle_weight'].val,
                    angular_velocity_weight=self.sliders['angular_velocity_weight'].val,
                    control_weight=self.sliders['control_weight'].val
                )
            elif self.controller_type == 'pole':
                self.controller.set_poles(
                    pole1=self.sliders['pole1'].val,
                    pole2=self.sliders['pole2'].val,
                    pole3=self.sliders['pole3'].val,
                    pole4=self.sliders['pole4'].val
                )

        # Update noise parameters if observer present
        if self.observer is not None and 'noise_x' in self.sliders:
            self.noise_std_x = self.sliders['noise_x'].val / 1000  # Convert mm to m
            self.noise_std_theta = np.radians(self.sliders['noise_theta'].val)  # Convert deg to rad

        # Reset controller and observer state
        if self.controller is not None:
            self.controller.reset()
        if self.observer is not None:
            self.observer.reset()

        # Re-run simulation
        from src.simulation import simulate
        results = simulate(
            self.initial_state,
            self.t_span,
            controller=self.controller,
            enable_air_drag=self.enable_air_drag,
            observer=self.observer,
            noise_std_x=self.noise_std_x,
            noise_std_theta=self.noise_std_theta
        )

        # Update data
        self.t = results['t']
        self.states = results['states']
        control = results['control']
        
        # Pad control to match states length
        if len(control) < len(self.states):
            control = np.concatenate([[0.0], control])
        self.control = control
        self.dforce = self._compute_dforce()

        # Reset animation to frame 0
        self.current_frame = 0

        # Update axis limits for new data
        self._compute_axis_limits()
        self.ax_anim.set_xlim(self.xlim)
        self.ax_anim.set_ylim(self.ylim)

        # Update theta plot limits
        theta_deg = np.degrees(self.states[:, 2])
        self.ax_theta.set_ylim(min(theta_deg.min(), -10) - 5, max(theta_deg.max(), 10) + 5)

        # Update x plot limits
        self.ax_x.set_ylim(self.states[:, 0].min() - 0.1, self.states[:, 0].max() + 0.1)

        # Update force plot limits
        force_data = self.control[:len(self.t)]
        force_margin = max(abs(force_data.min()), abs(force_data.max()), 1) * 0.1
        self.ax_force.set_ylim(force_data.min() - force_margin, force_data.max() + force_margin)

        # Update dF/dt plot limits
        dforce_margin = max(abs(self.dforce.min()), abs(self.dforce.max()), 1) * 0.1
        self.ax_dforce.set_ylim(self.dforce.min() - dforce_margin, self.dforce.max() + dforce_margin)

        # Clear the plot lines by removing and recreating them
        self.theta_line.remove()
        self.x_line.remove()
        self.force_line.remove()
        self.dforce_line.remove()
        self.theta_marker.remove()
        self.x_marker.remove()
        self.force_marker.remove()
        self.dforce_marker.remove()

        self.theta_line, = self.ax_theta.plot([], [], 'r-', linewidth=1.5)
        self.x_line, = self.ax_x.plot([], [], 'b-', linewidth=1.5)
        self.force_line, = self.ax_force.plot([], [], 'g-', linewidth=1.5)
        self.dforce_line, = self.ax_dforce.plot([], [], 'm-', linewidth=1.5)
        self.theta_marker, = self.ax_theta.plot([], [], 'ro', markersize=8)
        self.x_marker, = self.ax_x.plot([], [], 'bo', markersize=8)
        self.force_marker, = self.ax_force.plot([], [], 'go', markersize=8)
        self.dforce_marker, = self.ax_dforce.plot([], [], 'mo', markersize=8)

        # Reset cart and pendulum to initial position
        x0 = self.states[0, 0]
        theta0 = self.states[0, 2]
        f0 = self.control[0] if len(self.control) > 0 else 0

        cart_x = x0 - self.cart_width / 2
        cart_y = -self.cart_height / 2
        self.cart_patch.set_x(cart_x)
        self.cart_patch.set_y(cart_y)

        wheel_y = -self.cart_height / 2
        self.wheel_left.center = (x0 - self.cart_width / 3, wheel_y)
        self.wheel_right.center = (x0 + self.cart_width / 3, wheel_y)

        pivot_x = x0
        pivot_y = self.cart_height / 2
        self.pivot_point.center = (pivot_x, pivot_y)

        pend_x = pivot_x + self.pendulum_length * np.sin(theta0)
        pend_y = pivot_y + self.pendulum_length * np.cos(theta0)
        self.pendulum_line.set_data([pivot_x, pend_x], [pivot_y, pend_y])

        self.time_text.set_text(f't = 0.00 s\nx = {x0:.3f} m\nθ = {np.degrees(theta0):.1f}°\nF = {f0:.2f} N')

        # Force complete redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _init_animation(self):
        """Initialize animation elements."""
        self.cart_patch.set_x(0)
        self.cart_patch.set_y(0)
        self.wheel_left.center = (0, 0)
        self.wheel_right.center = (0, 0)
        self.pendulum_line.set_data([], [])
        self.pivot_point.center = (0, 0)
        self.time_text.set_text('')
        self.theta_line.set_data([], [])
        self.x_line.set_data([], [])
        self.force_line.set_data([], [])
        self.dforce_line.set_data([], [])
        self.theta_marker.set_data([], [])
        self.x_marker.set_data([], [])
        self.force_marker.set_data([], [])
        self.dforce_marker.set_data([], [])

        return (self.cart_patch, self.wheel_left, self.wheel_right,
                self.pendulum_line, self.pivot_point, self.time_text,
                self.theta_line, self.x_line, self.force_line, self.dforce_line,
                self.theta_marker, self.x_marker, self.force_marker, self.dforce_marker)
    
    def _update_animation(self, frame_idx):
        """Update animation for given frame."""
        # Use internal frame counter that resets on slider change
        frame = (self.current_frame * self.skip_frames) % len(self.t)
        self.current_frame += 1

        x = self.states[frame, 0]
        theta = self.states[frame, 2]
        t_current = self.t[frame]
        force_current = self.control[frame] if frame < len(self.control) else 0
        dforce_current = self.dforce[frame] if frame < len(self.dforce) else 0

        # Cart position
        cart_x = x - self.cart_width / 2
        cart_y = -self.cart_height / 2
        self.cart_patch.set_x(cart_x)
        self.cart_patch.set_y(cart_y)

        # Wheels
        wheel_radius = self.cart_height * 0.25
        wheel_y = -self.cart_height / 2
        self.wheel_left.center = (x - self.cart_width / 3, wheel_y)
        self.wheel_right.center = (x + self.cart_width / 3, wheel_y)

        # Pivot point
        pivot_x = x
        pivot_y = self.cart_height / 2
        self.pivot_point.center = (pivot_x, pivot_y)

        # Pendulum endpoint (theta=0 is upright, positive tilts right)
        pend_x = pivot_x + self.pendulum_length * np.sin(theta)
        pend_y = pivot_y + self.pendulum_length * np.cos(theta)
        self.pendulum_line.set_data([pivot_x, pend_x], [pivot_y, pend_y])

        # Time text
        theta_deg = np.degrees(theta)
        self.time_text.set_text(f't = {t_current:.2f} s\nx = {x:.3f} m\nθ = {theta_deg:.1f}°\nF = {force_current:.2f} N')

        # Update live plots
        t_data = self.t[:frame + 1]
        theta_data = np.degrees(self.states[:frame + 1, 2])
        x_data = self.states[:frame + 1, 0]
        force_data = self.control[:frame + 1]
        dforce_data = self.dforce[:frame + 1]

        self.theta_line.set_data(t_data, theta_data)
        self.x_line.set_data(t_data, x_data)
        self.force_line.set_data(t_data, force_data)
        self.dforce_line.set_data(t_data, dforce_data)

        self.theta_marker.set_data([t_current], [theta_deg])
        self.x_marker.set_data([t_current], [x])
        self.force_marker.set_data([t_current], [force_current])
        self.dforce_marker.set_data([t_current], [dforce_current])

        return (self.cart_patch, self.wheel_left, self.wheel_right,
                self.pendulum_line, self.pivot_point, self.time_text,
                self.theta_line, self.x_line, self.force_line, self.dforce_line,
                self.theta_marker, self.x_marker, self.force_marker, self.dforce_marker)
    
    def animate(self, interval=20, skip_frames=1, save_path=None):
        """Run the animation."""
        self._setup_figure()
        self.skip_frames = skip_frames
        
        # Use a generator that runs indefinitely
        def frame_generator():
            i = 0
            while True:
                yield i
                i += 1
        
        self.anim = FuncAnimation(
            self.fig,
            self._update_animation,
            frames=frame_generator(),
            init_func=self._init_animation,
            interval=interval,
            blit=True,
            cache_frame_data=False,
        )
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            self.anim.save(save_path, writer='pillow', fps=1000 // interval)
            print("Done.")
        
        plt.show()


def animate_from_file(filepath, **kwargs):
    """Load results from file and animate."""
    t, states, metadata = load_results(filepath)
    
    angle = metadata.get('initial_angle_deg', '?')
    title = f"Cart-Pendulum Animation (θ₀ = {angle}°)"
    
    animator = CartPendulumAnimator(t, states, title=title)
    animator.animate(**kwargs)


def animate_from_arrays(t, states, title="Cart-Pendulum", controller=None,
                        observer=None, show_sliders=False, initial_state=None,
                        t_span=None, enable_air_drag=True, control=None,
                        noise_std_x=0.002, noise_std_theta=0.005, **kwargs):
    """Animate from arrays with optional controller/observer for sliders."""
    animator = CartPendulumAnimator(
        t, states, title=title,
        controller=controller,
        observer=observer,
        show_sliders=show_sliders,
        initial_state=initial_state,
        t_span=t_span,
        enable_air_drag=enable_air_drag,
        control=control,
        noise_std_x=noise_std_x,
        noise_std_theta=noise_std_theta
    )
    animator.animate(**kwargs)