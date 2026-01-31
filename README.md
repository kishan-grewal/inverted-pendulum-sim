# Inverted Pendulum Simulation - COMP0216

**Team**: 4
**Date**: January 2026

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate.ps1
pip install -r requirements.txt
```

## Running Simulations

### Evaluation A: Disturbance Rejection
```bash
python main.py -A --controller lqr
python main.py -A --controller pole
```

### Evaluation B: Recovery from Starting Angle
```bash
python main.py -B 15 --controller lqr
python main.py -B 15 --controller pole
```

### Evaluation C: Sprint to 2m Target
```bash
python main.py -C --controller lqr
python main.py -C --controller pole
```

## Optional Flags

- `--no-observer` - Use perfect state feedback (no noise filtering)
- `--no-drag` - Disable air drag
- `--no-textboxes` - Disable interactive parameter tuning

## Code Structure

- `main.py` - Main evaluation script
- `src/dynamics.py` - Nonlinear and linearized dynamics
- `src/controllers.py` - LQR and PID controllers
- `src/observers.py` - Luenberger observer (noise filtering)
- `src/simulation.py` - ODE solver with sensor noise
- `src/visualisation.py` - Real-time animation
- `src/plots.py` - Performance plots

## Controllers Implemented

1. **LQR** - Linear Quadratic Regulator (energy-weighted optimal control)
2. **Pole Placement** - State feedback with direct eigenvalue assignment

## Sensor Noise Model

- Position noise: Gaussian, σ in metres
- Angle noise: Gaussian, σ in degrees
- Filtered using Luenberger observer with fixed poles