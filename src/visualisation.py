import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import TextBox

from src.dynamics import L_rod


# ===== TEXTBOX LAYOUT CONSTANTS =====
TEXTBOX_Y_START = 0.25
TEXTBOX_HEIGHT = 0.025
TEXTBOX_SPACING = 0.035
TEXTBOX_WIDTH = 0.15
TEXTBOX_X_POS = 0.10


class CartPendulumAnimator:
    """Animator for cart-pendulum system with optional controller and noise textboxes."""

    def __init__(self, t, states, cart_width=0.3, cart_height=0.15,
                 pendulum_length=None, title="Cart-Pendulum Animation",
                 controller=None, observer=None, show_textboxes=False,
                 initial_state=None, t_span=None, enable_air_drag=True,
                 control=None, noise_std_x=0.002, noise_std_theta=0.005,
                 disturbances=None):
        self.t = t
        self.states = states
        self.control = control if control is not None else np.zeros(len(t))
        self.cart_width = cart_width
        self.cart_height = cart_height
        self.pendulum_length = pendulum_length if pendulum_length else L_rod
        self.title = title
        self.controller = controller
        self.observer = observer
        self.show_textboxes = show_textboxes and (controller is not None or observer is not None)
        self.disturbances = disturbances

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
        self.ax_disturbance = None
        self.anim = None

        # Textbox references
        self.textboxes = {}
    
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

        if self.show_textboxes:
            self.fig = plt.figure(figsize=(20, 10))
        else:
            self.fig = plt.figure(figsize=(16, 9))
        self.fig.suptitle(self.title)

        # Adjust layout for textboxes if needed
        if self.show_textboxes:
            # Main animation axes
            self.ax_anim = self.fig.add_axes([0.03, 0.30, 0.38, 0.60])

            # Right side: 2x2 grid of plots
            self.ax_theta = self.fig.add_axes([0.46, 0.55, 0.24, 0.35])
            self.ax_force = self.fig.add_axes([0.74, 0.55, 0.24, 0.35])
            self.ax_x = self.fig.add_axes([0.46, 0.10, 0.24, 0.35])
            self.ax_disturbance = self.fig.add_axes([0.74, 0.10, 0.24, 0.35])

            # Create textboxes based on controller type
            self._setup_textboxes()

        else:
            # Standard layout without textboxes
            self.ax_anim = self.fig.add_axes([0.03, 0.10, 0.40, 0.80])
            self.ax_theta = self.fig.add_axes([0.50, 0.55, 0.22, 0.35])
            self.ax_force = self.fig.add_axes([0.76, 0.55, 0.22, 0.35])
            self.ax_x = self.fig.add_axes([0.50, 0.10, 0.22, 0.35])
            self.ax_disturbance = self.fig.add_axes([0.76, 0.10, 0.22, 0.35])

        self._setup_axes()
        self._setup_artists()
    
    def _setup_textboxes(self):
        """Create textboxes based on controller type and observer."""
        # TEXTBOX LAYOUT CONSTANTS are at top of file
        current_y = TEXTBOX_Y_START
        
        if self.controller_type == 'pid':
            info = self.controller.get_info()
            
            ax_kp = self.fig.add_axes([TEXTBOX_X_POS, current_y, TEXTBOX_WIDTH, TEXTBOX_HEIGHT])
            self.textboxes['Kp'] = TextBox(ax_kp, 'Kp', initial=str(info.get('Kp', 50.0)))
            self.textboxes['Kp'].on_submit(lambda text: self._on_textbox_submit('Kp', text))
            current_y -= TEXTBOX_SPACING
            
            ax_ki = self.fig.add_axes([TEXTBOX_X_POS, current_y, TEXTBOX_WIDTH, TEXTBOX_HEIGHT])
            self.textboxes['Ki'] = TextBox(ax_ki, 'Ki', initial=str(info.get('Ki', 0.0)))
            self.textboxes['Ki'].on_submit(lambda text: self._on_textbox_submit('Ki', text))
            current_y -= TEXTBOX_SPACING
            
            ax_kd = self.fig.add_axes([TEXTBOX_X_POS, current_y, TEXTBOX_WIDTH, TEXTBOX_HEIGHT])
            self.textboxes['Kd'] = TextBox(ax_kd, 'Kd', initial=str(info.get('Kd', 3.0)))
            self.textboxes['Kd'].on_submit(lambda text: self._on_textbox_submit('Kd', text))
            current_y -= TEXTBOX_SPACING
        
        elif self.controller_type == 'lqr':
            info = self.controller.get_info()
            weights = info.get('tuning_weights', {})
            
            ax_pos = self.fig.add_axes([TEXTBOX_X_POS, current_y, TEXTBOX_WIDTH, TEXTBOX_HEIGHT])
            self.textboxes['position_weight'] = TextBox(ax_pos, 'Pos Weight', 
                                                        initial=str(weights.get('position_weight', 1.0)))
            self.textboxes['position_weight'].on_submit(lambda text: self._on_textbox_submit('position_weight', text))
            current_y -= TEXTBOX_SPACING
            
            ax_vel = self.fig.add_axes([TEXTBOX_X_POS, current_y, TEXTBOX_WIDTH, TEXTBOX_HEIGHT])
            self.textboxes['velocity_weight'] = TextBox(ax_vel, 'Vel Weight', 
                                                        initial=str(weights.get('velocity_weight', 1.0)))
            self.textboxes['velocity_weight'].on_submit(lambda text: self._on_textbox_submit('velocity_weight', text))
            current_y -= TEXTBOX_SPACING
            
            ax_ang = self.fig.add_axes([TEXTBOX_X_POS, current_y, TEXTBOX_WIDTH, TEXTBOX_HEIGHT])
            self.textboxes['angle_weight'] = TextBox(ax_ang, 'Ang Weight', 
                                                    initial=str(weights.get('angle_weight', 1.0)))
            self.textboxes['angle_weight'].on_submit(lambda text: self._on_textbox_submit('angle_weight', text))
            current_y -= TEXTBOX_SPACING
            
            ax_angvel = self.fig.add_axes([TEXTBOX_X_POS, current_y, TEXTBOX_WIDTH, TEXTBOX_HEIGHT])
            self.textboxes['angular_velocity_weight'] = TextBox(ax_angvel, 'AngVel Weight',
                                                                initial=str(weights.get('angular_velocity_weight', 1.0)))
            self.textboxes['angular_velocity_weight'].on_submit(lambda text: self._on_textbox_submit('angular_velocity_weight', text))
            current_y -= TEXTBOX_SPACING
            
            ax_ctrl = self.fig.add_axes([TEXTBOX_X_POS, current_y, TEXTBOX_WIDTH, TEXTBOX_HEIGHT])
            self.textboxes['control_weight'] = TextBox(ax_ctrl, 'Ctrl Weight', 
                                                        initial=str(weights.get('control_weight', 1.0)))
            self.textboxes['control_weight'].on_submit(lambda text: self._on_textbox_submit('control_weight', text))
            current_y -= TEXTBOX_SPACING
        
        elif self.controller_type == 'pole':
            info = self.controller.get_info()
            poles = info.get('desired_poles', [-2, -3, -4, -5])
            poles_str = ', '.join(str(p) for p in poles)
            
            ax_poles = self.fig.add_axes([TEXTBOX_X_POS, current_y, TEXTBOX_WIDTH, TEXTBOX_HEIGHT])
            self.textboxes['poles'] = TextBox(ax_poles, 'Poles', initial=poles_str)
            self.textboxes['poles'].on_submit(lambda text: self._on_textbox_submit('poles', text))
            current_y -= TEXTBOX_SPACING
        
        if self.observer is not None:
            ax_noise_x = self.fig.add_axes([TEXTBOX_X_POS, current_y, TEXTBOX_WIDTH, TEXTBOX_HEIGHT])
            self.textboxes['noise_x'] = TextBox(ax_noise_x, 'Noise σ_x [m]', 
                                                initial=str(self.noise_std_x))
            self.textboxes['noise_x'].on_submit(lambda text: self._on_textbox_submit('noise_x', text))
            current_y -= TEXTBOX_SPACING
            
            ax_noise_theta = self.fig.add_axes([TEXTBOX_X_POS, current_y, TEXTBOX_WIDTH, TEXTBOX_HEIGHT])
            self.textboxes['noise_theta'] = TextBox(ax_noise_theta, 'Noise σ_θ [°]',
                                                    initial=str(self.noise_std_theta))
            self.textboxes['noise_theta'].on_submit(lambda text: self._on_textbox_submit('noise_theta', text))
        
        self.fig.text(0.03, 0.01,
                    "Enter values and press Enter to re-run simulation",
                    fontsize=9, style='italic', alpha=0.7)

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

        # Configure disturbance plot
        self.ax_disturbance.set_xlim(0, self.t[-1])
        self.ax_disturbance.set_xlabel('Time [s]')
        self.ax_disturbance.grid(True, alpha=0.3)
        
        if self.disturbances is not None and len(self.disturbances) > 0:
            # Determine if we're showing cart or angular disturbance
            first_disturbance = self.disturbances[0]
            _, cart_impulse, angular_impulse = first_disturbance
            
            if angular_impulse != 0:
                # Pendulum disturbance
                self.ax_disturbance.set_ylabel('Angular [N·s·m]')
                self.ax_disturbance.set_title('Pendulum Disturbance')
                self.ax_disturbance.set_ylim(-0.05, 0.10)
            else:
                # Cart disturbance
                self.ax_disturbance.set_ylabel('Cart [N·s]')
                self.ax_disturbance.set_title('Cart Disturbance')
                self.ax_disturbance.set_ylim(-0.5, 4.0)
        else:
            # No disturbance
            self.ax_disturbance.set_ylabel('Force')
            self.ax_disturbance.set_title('No Disturbance')
            self.ax_disturbance.set_ylim(-1, 1)
    
    def _setup_artists(self):
        """Create all plot artists."""
        # Cart
        self.cart_patch = FancyBboxPatch(
            (0, 0), self.cart_width, self.cart_height,
            boxstyle="round,pad=0.01",
            facecolor='#4A4A4A',
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
        self.pendulum_line, = self.ax_anim.plot([], [], 'o-', color="#CE8D00",
                                                  linewidth=4, markersize=12,
                                                  markerfacecolor="#CE8D00")

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
        
        # Disturbance visualization
        self.disturbance_lines = []
        if self.disturbances is not None:
            for dist_time, cart_impulse, angular_impulse in self.disturbances:
                if angular_impulse != 0:
                    # Vertical line from 0 to impulse magnitude
                    line = self.ax_disturbance.vlines(x=dist_time, ymin=0, ymax=angular_impulse,
                                                      colors='red', linewidth=3, alpha=0.7,
                                                      label=f'{angular_impulse} N·s·m @ {dist_time:.1f}s')
                else:
                    # Vertical line from 0 to impulse magnitude
                    line = self.ax_disturbance.vlines(x=dist_time, ymin=0, ymax=cart_impulse,
                                                      colors='blue', linewidth=3, alpha=0.7,
                                                      label=f'{cart_impulse} N·s @ {dist_time:.1f}s')
                self.disturbance_lines.append(line)
            self.ax_disturbance.legend(loc='upper right', fontsize=8)

        # Current position markers
        self.theta_marker, = self.ax_theta.plot([], [], 'ro', markersize=8)
        self.x_marker, = self.ax_x.plot([], [], 'bo', markersize=8)
        self.force_marker, = self.ax_force.plot([], [], 'go', markersize=8)
    
    def _on_textbox_submit(self, param_name, text):
        try:
            value = float(text)
        except ValueError:
            # Check if this is the poles parameter (comma-separated)
            if param_name == 'poles':
                try:
                    poles = [float(p.strip()) for p in text.split(',')]
                    if len(poles) != 4:
                        print(f"Error: Expected 4 poles, got {len(poles)}")
                        return
                    # Successfully parsed 4 poles
                    self.controller.set_poles(pole1=poles[0], pole2=poles[1], 
                                            pole3=poles[2], pole4=poles[3])
                except (ValueError, AttributeError) as e:
                    print(f"Invalid pole input: '{text}' (expected format: -2, -3, -4, -5)")
                    return
            else:
                print(f"Invalid input for {param_name}: '{text}' (must be a number)")
                return

        # Handle noise parameters FIRST (before controller checks)
        if param_name == 'noise_x':
            self.noise_std_x = value
        elif param_name == 'noise_theta':
            self.noise_std_theta = value
        
        # Then handle controller parameters (skip poles since handled above)
        elif self.controller is not None and param_name != 'poles':
            if self.controller_type == 'pid':
                if param_name == 'Kp':
                    self.controller.set_gains(Kp=value)
                elif param_name == 'Ki':
                    self.controller.set_gains(Ki=value)
                elif param_name == 'Kd':
                    self.controller.set_gains(Kd=value)
            
            elif self.controller_type == 'lqr':
                kwargs = {param_name: value}
                self.controller.set_weights(**kwargs)

        # Reset and re-run simulation
        if self.controller is not None:
            self.controller.reset()
        if self.observer is not None:
            self.observer.reset()

        from src.simulation import simulate
        results = simulate(
            self.initial_state,
            self.t_span,
            controller=self.controller,
            enable_air_drag=self.enable_air_drag,
            observer=self.observer,
            noise_std_x=self.noise_std_x,
            noise_std_theta=self.noise_std_theta,
            disturbances=self.disturbances
        )

        self.t = results['t']
        self.states = results['states']
        control = results['control']
        
        if len(control) < len(self.states):
            control = np.concatenate([[0.0], control])
        self.control = control

        self.current_frame = 0

        self._compute_axis_limits()
        self.ax_anim.set_xlim(self.xlim)
        self.ax_anim.set_ylim(self.ylim)

        theta_deg = np.degrees(self.states[:, 2])
        self.ax_theta.set_ylim(min(theta_deg.min(), -10) - 5, max(theta_deg.max(), 10) + 5)

        self.ax_x.set_ylim(self.states[:, 0].min() - 0.1, self.states[:, 0].max() + 0.1)

        force_data = self.control[:len(self.t)]
        force_margin = max(abs(force_data.min()), abs(force_data.max()), 1) * 0.1
        self.ax_force.set_ylim(force_data.min() - force_margin, force_data.max() + force_margin)

        self.theta_line.remove()
        self.x_line.remove()
        self.force_line.remove()
        self.theta_marker.remove()
        self.x_marker.remove()
        self.force_marker.remove()

        self.theta_line, = self.ax_theta.plot([], [], 'r-', linewidth=1.5)
        self.x_line, = self.ax_x.plot([], [], 'b-', linewidth=1.5)
        self.force_line, = self.ax_force.plot([], [], 'g-', linewidth=1.5)
        self.theta_marker, = self.ax_theta.plot([], [], 'ro', markersize=8)
        self.x_marker, = self.ax_x.plot([], [], 'bo', markersize=8)
        self.force_marker, = self.ax_force.plot([], [], 'go', markersize=8)

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
        self.theta_marker.set_data([], [])
        self.x_marker.set_data([], [])
        self.force_marker.set_data([], [])

        return (self.cart_patch, self.wheel_left, self.wheel_right,
                self.pendulum_line, self.pivot_point, self.time_text,
                self.theta_line, self.x_line, self.force_line,
                self.theta_marker, self.x_marker, self.force_marker)
    
    def _update_animation(self, frame_idx):
        """Update animation for given frame."""
        # Use internal frame counter that resets on textbox change
        frame = (self.current_frame * self.skip_frames) % len(self.t)
        self.current_frame += 1

        x = self.states[frame, 0]
        theta = self.states[frame, 2]
        t_current = self.t[frame]
        force_current = self.control[frame] if frame < len(self.control) else 0

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

        self.theta_line.set_data(t_data, theta_data)
        self.x_line.set_data(t_data, x_data)
        self.force_line.set_data(t_data, force_data)

        self.theta_marker.set_data([t_current], [theta_deg])
        self.x_marker.set_data([t_current], [x])
        self.force_marker.set_data([t_current], [force_current])

        return (self.cart_patch, self.wheel_left, self.wheel_right,
                self.pendulum_line, self.pivot_point, self.time_text,
                self.theta_line, self.x_line, self.force_line,
                self.theta_marker, self.x_marker, self.force_marker)
    
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


def animate_from_arrays(t, states, title="Cart-Pendulum", controller=None,
                        observer=None, show_textboxes=False, initial_state=None,
                        t_span=None, enable_air_drag=True, control=None,
                        noise_std_x=0.002, noise_std_theta=0.005, 
                        disturbances=None, **kwargs):
    """Animate from arrays with optional controller/observer for textboxes."""
    animator = CartPendulumAnimator(
        t, states, title=title,
        controller=controller,
        observer=observer,
        show_textboxes=show_textboxes,
        initial_state=initial_state,
        t_span=t_span,
        enable_air_drag=enable_air_drag,
        control=control,
        noise_std_x=noise_std_x,
        noise_std_theta=noise_std_theta,
        disturbances=disturbances
    )
    animator.animate(**kwargs)