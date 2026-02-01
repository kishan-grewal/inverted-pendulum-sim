(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -A --controller lqr
Evaluation A
Controller: LQR

Results:
  Max angle deviation: 5.07°
  Settling time: 1.11s
  Status: SUCCESS
  Max control force: 3.63 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -A --controller pole
Evaluation A
Controller: POLE

Results:
  Max angle deviation: 5.91°
  Settling time: 1.85s
  Status: SUCCESS
  Max control force: 3.34 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -B 15 --controller lqr
Evaluation B (15.0°)
Controller: LQR

Results:
  Max angle deviation: 15.00°
  Stabilised: Yes
  Settling time: 2.33s
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -B 15 --controller pole
Evaluation B (15.0°)
Controller: POLE

Results:
  Max angle deviation: 15.00°
  Stabilised: Yes
  Settling time: 2.28s
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -C --controller lqr
Evaluation C (Sprint)
Controller: LQR

Results:
  Time to reach 2.0m: 2.03s
  Final position: 2.002m (error: 0.002m)
  Final velocity: 0.041m/s
  Max angle deviation: 22.11°
  Status: SUCCESS
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -C --controller lqr
Evaluation C (Sprint)
Controller: LQR

Results:
  Time to reach 2.0m: 1.97s
  Final position: 1.991m (error: 0.009m)
  Final velocity: 0.024m/s
  Max angle deviation: 22.25°
  Status: SUCCESS
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -C --controller pole
Evaluation C (Sprint)
Controller: POLE

Results:
  Time to reach 2.0m: 1.43s
  Final position: 1.997m (error: 0.003m)
  Final velocity: 0.031m/s
  Max angle deviation: 26.19°
  Status: SUCCESS
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -B 15 --controller lqr  # with noise (TOP LEFT)
Evaluation B (15.0°)
Controller: LQR

Results:
  Max angle deviation: 15.00°
  Stabilised: Yes
  Settling time: 1.88s
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -B 15 --controller lqr --no-observer  # without noise (TOP RIGHT)
Evaluation B (15.0°)
Controller: LQR

Results:
  Max angle deviation: 15.00°
  Stabilised: Yes
  Settling time: 1.75s
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -B 15 --controller lqr --no-drag  # without drag (BOTTOM LEFT)
Evaluation B (15.0°)
Controller: LQR

Results:
  Max angle deviation: 15.00°
  Stabilised: Yes
  Settling time: 1.66s
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -B 15 --controller lqr --no-drag --no-observer  # without drag or noise (BOTTOM RIGHT)
Evaluation B (15.0°)
Controller: LQR

Results:
  Max angle deviation: 15.00°
  Stabilised: Yes
  Settling time: 1.71s
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -C --controller lqr 
Evaluation C (Sprint)
Controller: LQR

Results:
  Time to reach 2.0m: 1.95s
  Final position: 1.996m (error: 0.004m)
  Final velocity: 0.033m/s
  Max angle deviation: 22.29°
  Status: SUCCESS
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -C --controller pole
Evaluation C (Sprint)
Controller: POLE

Results:
  Time to reach 2.0m: 1.38s
  Final position: 1.992m (error: 0.008m)
  Final velocity: 0.053m/s
  Max angle deviation: 26.61°
  Status: SUCCESS
  Max control force: 15.00 N

(.venv) PS C:\Users\Kishan\Documents\Semester2\SystemsEngineering\inverted-pendulum-sim> python main.py -A --controller lqr 
Evaluation A
Controller: LQR

Results:
  Max angle deviation: 5.21°
  Settling time: 2.05s
  Status: SUCCESS
  Max control force: 5.78 N
















