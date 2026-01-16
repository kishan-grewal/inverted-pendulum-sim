# inverted-pendulum-cart

## TWO ALGORITHMS PID + LQR, DO AT LEAST TASK 2 (RECOVERY FROM STARTING ANGLE)

### Setup:
```bash
python -m venv .venv
.venv/Scripts/activate.ps1
pip install -r requirements.txt
```

### Example of Conventional Commits:
- feat: add passive drop simulation
- fix: correct sign error in pendulum torque
- docs: explain simulation output format
- chore: add numpy and scipy dependencies

Overview:
The rover is modelled as a cart-pendulum system with a single equivalent longitudinal force input representing the combined effect of wheels and motors.

Design space overview:

Controllers:
- **LQR** - linear quadratic state feedback from linearised model
- **MPC** - model predictive control with explicit constraint handling

State estimation:
- **Filtered differentiation** - numerical differentiation with low-pass filtering
- **Luenberger observer** - deterministic model-based state observer
- **Kalman filter** - stochastic observer with explicit noise modelling

Controller-estimator combinations are evaluated for robustness, performance limits, and implementation risk using identical plants and test scenarios.

Model and simulation decisions:
- Continuous-time nonlinear plant
- State: [x, x_dot, theta, theta_dot]
- Theta = 0 at upright equilibrium
- Single horizontal force input F
- No drivetrain, wheel slip, or motor dynamics
- No discrete sampling or ZOH modelling
- Adaptive ODE integration

Controllers:
- Angle-only PID (baseline)
- Full-state feedback controller (LQR or MPC under evaluation)
- Same plant, disturbances, and tests for all controllers

Disturbances and tests:
- Initial angle offsets
- Impulse forces on cart
- Position reference steps (sprint and stop)

Libraries:

Core numerics:
- numpy
- scipy.integrate.solve_ivp
- (odeint legacy only don't use)

Control:
- control (python-control)
- PID implemented manually

Noise and filtering:
- numpy.random
- scipy.signal

Visualisation:
- matplotlib.pyplot
- matplotlib.animation.FuncAnimation
- matplotlib.widgets

Real-time animation:
- matplotlib (default, 2D)
- vpython (optional, 3D)
- pygame (optional, 2D game-loop)

Scope:
Focus is on control performance comparison and robustness. Control and estimation approaches are selected based on simulation evidence rather than theoretical optimality. Low-level motor, drivetrain, and real-time OS modelling are out of scope.