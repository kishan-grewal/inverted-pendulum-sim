```markdown
# Inverted Pendulum Simulation - Recording Plan

## 5-Slide Structure

### Slide 1: Eval A - Disturbance Rejection (2 GIFs)
```bash
python main.py -A --controller lqr
python main.py -A --controller pole
```
**Shows:** Both controllers meet stabilization requirement

### Slide 2: Eval B - 15° Recovery (2 GIFs)
```bash
python main.py -B 15 --controller lqr
python main.py -B 15 --controller pole
```
**Shows:** Both controllers meet recovery requirement

### Slide 3: Eval C - Sprint (2 GIFs)
```bash
python main.py -C --controller lqr
python main.py -C --controller pole
```
**Shows:** Both controllers meet sprint requirement

### Slide 4: Noise Comparison - Eval B (2 GIFs)
```bash
python main.py -B 15 --controller lqr              # with noise (baseline)
python main.py -B 15 --controller lqr --no-observer  # without noise
```
**Shows:** System handles sensor noise via observer

### Slide 5: Drag Comparison - Eval B (2 GIFs)
```bash
python main.py -B 15 --controller lqr           # with drag (baseline)
python main.py -B 15 --controller lqr --no-drag   # without drag
```
**Shows:** System behavior with/without physical damping

---

## Why 5 Slides?

**Slides 1-3:** Prove all three requirements met (stabilize, recover, sprint) with both controllers

**Slides 4-5:** Show ablation studies (noise, drag) as required by spec - demonstrates understanding of physical effects and proper system modeling

**Clean comparison structure:** Each slide shows one clear contrast, easy to understand side-by-side

**Total: 10 unique GIF recordings**
```