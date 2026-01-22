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
    """Animator for cart-pendulum system with optional PID gain sliders."""
    
    def __init__(self, t, states, cart_width=0.3, cart_height=0.15, 
                 pendulum_length=None, title="Cart-Pendulum Animation",
                 controller=None, show_sliders=False):
        self.t = t
        self.states = states
        self.cart_width = cart_width
        self.cart_height = cart_height
        self.pendulum_length = pendulum_length if pendulum_length else L_rod
        self.title = title
        self.controller = controller
        self.show_sliders = show_sliders and controller is not None
        
        # Compute axis limits from data
        x_min = np.min(states[:, 0]) - cart_width - self.pendulum_length
        x_max = np.max(states[:, 0]) + cart_width + self.pendulum_length
        y_min = -self.pendulum_length - cart_height
        y_max = self.pendulum_length + cart_height
        
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        self.xlim = (x_min - x_padding, x_max + x_padding)
        self.ylim = (y_min - y_padding, y_max + y_padding)
        
        self.fig = None
        self.ax_anim = None
        self.ax_theta = None
        self.ax_x = None
        self.anim = None
        
        # Slider references
        self.slider_kp = None
        self.slider_ki = None
        self.slider_kd = None
    
    def _setup_figure(self):
        """Create figure with animation and plots."""
        if self.show_sliders:
            self.fig = plt.figure(figsize=(16, 10))
        else:
            self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle(self.title)
        
        # Adjust layout for sliders if needed
        if self.show_sliders:
            # Main animation axes
            self.ax_anim = self.fig.add_axes([0.05, 0.25, 0.45, 0.65])
            
            # Theta plot
            self.ax_theta = self.fig.add_axes([0.58, 0.55, 0.38, 0.35])
            
            # X position plot
            self.ax_x = self.fig.add_axes([0.58, 0.1, 0.38, 0.35])
            
            # Slider axes
            ax_kp = self.fig.add_axes([0.15, 0.15, 0.3, 0.03])
            ax_ki = self.fig.add_axes([0.15, 0.10, 0.3, 0.03])
            ax_kd = self.fig.add_axes([0.15, 0.05, 0.3, 0.03])
            
            # Get current gains from controller
            info = self.controller.get_info()
            current_kp = info.get('Kp', 50.0)
            current_ki = info.get('Ki', 0.5)
            current_kd = info.get('Kd', 15.0)
            
            # Create sliders
            self.slider_kp = Slider(ax_kp, 'Kp', 0.0, 200.0, valinit=current_kp, valstep=1.0)
            self.slider_ki = Slider(ax_ki, 'Ki', 0.0, 10.0, valinit=current_ki, valstep=0.1)
            self.slider_kd = Slider(ax_kd, 'Kd', 0.0, 50.0, valinit=current_kd, valstep=0.5)
            
            # Connect slider callbacks
            self.slider_kp.on_changed(self._update_gains)
            self.slider_ki.on_changed(self._update_gains)
            self.slider_kd.on_changed(self._update_gains)
            
            # Add note about sliders
            self.fig.text(0.05, 0.01, 
                         "Note: Sliders update controller gains for next simulation run",
                         fontsize=9, style='italic', alpha=0.7)
        else:
            # Standard layout without sliders
            self.ax_anim = self.fig.add_axes([0.05, 0.15, 0.5, 0.75])
            self.ax_theta = self.fig.add_axes([0.62, 0.55, 0.35, 0.35])
            self.ax_x = self.fig.add_axes([0.62, 0.1, 0.35, 0.35])
        
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
        
        # Current position markers
        self.theta_marker, = self.ax_theta.plot([], [], 'ro', markersize=8)
        self.x_marker, = self.ax_x.plot([], [], 'bo', markersize=8)
    
    def _update_gains(self, val):
        """Callback for slider changes."""
        if self.controller is not None and hasattr(self.controller, 'set_gains'):
            self.controller.set_gains(
                Kp=self.slider_kp.val,
                Ki=self.slider_ki.val,
                Kd=self.slider_kd.val
            )
    
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
        self.theta_marker.set_data([], [])
        self.x_marker.set_data([], [])
        
        return (self.cart_patch, self.wheel_left, self.wheel_right,
                self.pendulum_line, self.pivot_point, self.time_text,
                self.theta_line, self.x_line, self.theta_marker, self.x_marker)
    
    def _update_animation(self, frame):
        """Update animation for given frame."""
        x = self.states[frame, 0]
        theta = self.states[frame, 2]
        t_current = self.t[frame]
        
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
        self.time_text.set_text(f't = {t_current:.2f} s\nx = {x:.3f} m\nθ = {theta_deg:.1f}°')
        
        # Update live plots
        t_data = self.t[:frame + 1]
        theta_data = np.degrees(self.states[:frame + 1, 2])
        x_data = self.states[:frame + 1, 0]
        
        self.theta_line.set_data(t_data, theta_data)
        self.x_line.set_data(t_data, x_data)
        
        self.theta_marker.set_data([t_current], [theta_deg])
        self.x_marker.set_data([t_current], [x])
        
        return (self.cart_patch, self.wheel_left, self.wheel_right,
                self.pendulum_line, self.pivot_point, self.time_text,
                self.theta_line, self.x_line, self.theta_marker, self.x_marker)
    
    def animate(self, interval=20, skip_frames=1, save_path=None):
        """Run the animation."""
        self._setup_figure()
        
        frames = range(0, len(self.t), skip_frames)
        
        self.anim = FuncAnimation(
            self.fig,
            self._update_animation,
            frames=frames,
            init_func=self._init_animation,
            interval=interval,
            blit=True,
            repeat=True,
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
                        show_sliders=False, **kwargs):
    """Animate from arrays with optional controller for sliders."""
    animator = CartPendulumAnimator(t, states, title=title, 
                                     controller=controller, 
                                     show_sliders=show_sliders)
    animator.animate(**kwargs)