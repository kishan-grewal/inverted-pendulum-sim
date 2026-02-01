(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -A --controller lqr
Evaluation A
Controller: LQR

Results:
  Max angle deviation: 5.08°
  Settling time: 1.90s
  Status: SUCCESS
  Max control force: 4.80 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -A --controller pole
Evaluation A
Controller: POLE

Results:
  Max angle deviation: 5.70°
  Settling time: 1.65s
  Status: SUCCESS
  Max control force: 4.82 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -B 15 --controller lqr
Evaluation B (15.0°)
Controller: LQR

Results:
  Max angle deviation: 15.00°
  Stabilised: Yes
  Settling time: 1.97s
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -B 15 --controller pole
Evaluation B (15.0°)
Controller: POLE

Results:
  Max angle deviation: 15.00°
  Stabilised: Yes
  Settling time: 2.08s
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -C --controller lqr
Evaluation C (Sprint)
Controller: LQR

Results:
  Time to reach 2.0m: 2.01s
  Final position: 2.006m (error: 0.006m)
  Final velocity: 0.025m/s
  Max angle deviation: 22.20°
  Status: SUCCESS
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -C --controller pole
Evaluation C (Sprint)
Controller: POLE

Results:
  Time to reach 2.0m: 1.39s
  Final position: 1.989m (error: 0.011m)
  Final velocity: 0.037m/s
  Max angle deviation: 26.75°
  Status: SUCCESS
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -B 15 --controller lqr --no-observer  # without noise
Evaluation B (15.0°)
Controller: LQR

Results:
  Max angle deviation: 15.00°
  Stabilised: Yes
  Settling time: 1.75s
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -B 15 --controller lqr --no-drag   # without drag
Evaluation B (15.0°)
Controller: LQR

Results:
  Max angle deviation: 15.00°
  Stabilised: No
  Settling time: 2.03s
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> 