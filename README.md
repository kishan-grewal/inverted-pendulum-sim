# inverted-pendulum-sim

## Checkout current branch
```bash
git fetch origin
git checkout add-evalC
```

## Setup:
```bash
python -m venv .venv
.venv/Scripts/activate.ps1
pip install -r requirements.txt
```

## Run:
```bash
python main.py -A --controller lqr
python main.py -B --controller lqr
```

## Example of Conventional Commits:
- feat: add passive drop simulation
- fix: correct sign error in pendulum torque
- docs: explain simulation output format
- chore: add numpy and scipy dependencies
