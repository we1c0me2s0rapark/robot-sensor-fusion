# Robot Sensor Fusion

EKF-based sensor fusion for a wheeled robot, fusing IMU and wheel odometry
to estimate position and velocity in real time.

## Requirements

- Python 3.9+
- numpy, matplotlib, pytest (see `requirements.txt`)

```bash
pip install -r requirements.txt
```

## Project structure

```
robot-sensor-fusion/
├── src/
│   ├── robot.py      # Unicycle kinematic model (ground truth)
│   ├── sensors.py    # Virtual IMU and wheel odometry with tunable noise
│   └── ekf.py        # Extended Kalman Filter
├── tests/
│   └── test_sensor_fusion.py
├── plots/            # Generated simulation outputs
├── main.py           # Simulation harness and evaluation scenarios
└── requirements.txt
```

## Running the demos

```bash
# All scenarios + performance benchmark
python main.py

# Performance benchmark only
python main.py --benchmark

# Specific scenario
python main.py --scenario cheap    # high-quality vs cheap IMU
python main.py --scenario fusion   # full fusion vs IMU-only dead-reckoning
```

Plots are saved to `plots/`.

## Running the tests

```bash
pytest tests/ -v
```

## Design overview

### State vector

```
x = [px, py, theta, v, omega]
```

| Symbol | Meaning             | Unit  |
|--------|---------------------|-------|
| px, py | Position            | m     |
| theta  | Heading             | rad   |
| v      | Linear velocity     | m/s   |
| omega  | Angular velocity    | rad/s |

### Algorithm: Extended Kalman Filter

An EKF was chosen over a UKF or particle filter for three reasons:

1. **Speed.** At 1 kHz the predict step must complete in under 1 ms.
   The analytic 5x5 Jacobian in the EKF adds negligible cost on top of
   the matrix multiply.  Typical measured cost: ~5 µs per call.

2. **The nonlinearity is mild.** The unicycle model has a single
   trigonometric nonlinearity (heading coupling into x/y).  A first-order
   Taylor expansion (EKF) handles this well.

3. **Commercial-friendliness.** No GPL dependencies -- pure NumPy.

### Sensor models

**IMUSensor** (`sensors.py`)

Noise model:

```
measurement = true_value + N(0, noise_std) + bias
bias(t+dt)  = bias(t) + N(0, bias_drift * sqrt(dt))
```

| Preset        | gyro_noise_std | gyro_bias_drift |
|---------------|----------------|-----------------|
| High-quality  | 0.001 rad/s    | 0.00001 rad/s/s |
| Cheap         | 0.05 rad/s     | 0.01 rad/s/s    |

**WheelOdometrySensor** (`sensors.py`)

Adds Gaussian noise to each wheel velocity independently.
Optional slip events (configurable probability) zero out one wheel.

### Predict vs update rates

| Step    | Trigger        | Typical rate |
|---------|----------------|--------------|
| predict | IMU reading    | 1 kHz        |
| update  | Odometry tick  | 50 Hz        |

The predict step propagates both the state and covariance using the
IMU as a control input.  The update step corrects accumulated IMU drift
using the wheel-odometry estimate of [v, omega].

### Performance

All EKF matrices are pre-allocated at construction time.
No heap allocation occurs in `predict()` or `update_odometry()`.
The 2x2 innovation covariance inversion in the update step uses the
explicit analytic inverse rather than `np.linalg.inv`.

Typical measured throughput on a standard laptop CPU:

| Operation | Time     |
|-----------|----------|
| predict() | ~5 µs    |
| update()  | ~8 µs    |

Both are well within the 1 kHz (1000 µs) budget.
