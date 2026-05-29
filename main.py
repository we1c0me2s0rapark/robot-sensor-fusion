"""
@file main.py
@brief Test harness for IMU and wheel odometry EKF sensor fusion.

This file runs multiple simulation scenarios to evaluate an Extended Kalman Filter
(EKF) for robot state estimation using IMU and wheel odometry sensors.

@details
The simulation supports three main use cases:
1. Comparison of high-quality versus low-quality IMU sensors to demonstrate drift behaviour.
2. Comparison of IMU-only estimation versus full sensor fusion using odometry.
3. Performance benchmarking of EKF predict and update steps at 1 kHz.

The system simulates a mobile robot moving in a 2D plane and evaluates estimation
accuracy against a known ground truth trajectory.
"""

from __future__ import annotations

import argparse
import sys
import time
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend, suitable for headless execution
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Make source directory importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.robot import Robot
from src.sensors import IMUSensor, WheelOdometrySensor
from src.ekf import EKF


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    @brief Simulation configuration parameters.

    @details
    Contains global timing and system parameters for the simulation environment.
    """
    duration: float = 30.0      # Simulation duration in seconds
    imu_rate: float = 1000.0    # IMU sampling frequency in Hz
    odom_rate: float = 50.0     # Odometry update frequency in Hz
    wheel_base: float = 0.5     # Distance between wheels in metres
    seed: int = 42              # Random seed for reproducibility


@dataclass
class IMUConfig:
    """
    @brief Noise parameters for an IMU preset.

    @details
    Defines noise characteristics and bias drift parameters for IMU simulation.
    """
    label:            str
    accel_noise_std:  float
    gyro_noise_std:   float
    accel_bias_drift: float
    gyro_bias_drift:  float


# ---------------------------------------------------------------------------
# IMU presets
# ---------------------------------------------------------------------------

IMU_HIGH_QUALITY = IMUConfig(
    label="High-quality IMU",
    accel_noise_std=0.01,
    gyro_noise_std=0.001,
    accel_bias_drift=0.0001,
    gyro_bias_drift=0.00001,
)

IMU_CHEAP = IMUConfig(
    label="Cheap IMU (high drift)",
    accel_noise_std=0.5,
    gyro_noise_std=0.05,
    accel_bias_drift=0.05,
    gyro_bias_drift=0.01,
)


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def compute_velocity_command(t: float) -> tuple[float, float]:
    """
    @brief Compute velocity commands for the robot.

    @param t Current simulation time in seconds
    @return Tuple containing linear velocity (m/s) and angular velocity (rad/s)

    @details
    Generates a smooth figure of eight-like motion using sinusoidal angular velocity.
    """
    v = 1.0                                     # constant forward speed
    omega = 0.5 * np.sin(2 * np.pi * t / 20.0)  # gentle sinusoidal turning
    return v, omega


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """
    @brief Stores time-series results from a simulation run.

    @details
    Contains both ground truth and estimated states for evaluation.
    """
    label: str
    times: list = field(default_factory=list)
    true_x: list = field(default_factory=list)
    true_y: list = field(default_factory=list)
    est_x: list = field(default_factory=list)
    est_y: list = field(default_factory=list)
    pos_error: list = field(default_factory=list)
    vel_error: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_simulation(
    imu_cfg: IMUConfig,
    sim_cfg: SimConfig,
    use_odometry: bool = True,
) -> SimResult:
    """
    @brief Run a full simulation for a given sensor configuration.

    @param imu_cfg IMU noise and bias configuration
    @param sim_cfg Global simulation configuration
    @param use_odometry If false, only IMU is used in EKF updates

    @return SimResult containing logged ground truth and estimates

    @details
    Executes a 1 kHz simulation loop with optional odometry updates.
    """

    robot = Robot(x=0.0, y=0.0, theta=0.0)

    imu = IMUSensor(
        accel_noise_std=imu_cfg.accel_noise_std,
        gyro_noise_std=imu_cfg.gyro_noise_std,
        accel_bias_drift=imu_cfg.accel_bias_drift,
        gyro_bias_drift=imu_cfg.gyro_bias_drift,
        rng=np.random.default_rng(sim_cfg.seed),
    )

    odom = WheelOdometrySensor(
        wheel_base=sim_cfg.wheel_base,
        noise_std=0.02,
        slip_probability=0.001,
        rng=np.random.default_rng(sim_cfg.seed + 1),
    )

    ekf = EKF(
        process_noise_std=(0.01, 0.01, 0.001, 0.1, 0.01),
        odom_noise_std=(0.05, 0.01),
    )

    label = imu_cfg.label
    if not use_odometry:
        label += " (no odometry)"

    result = SimResult(label=label)

    dt = 1.0 / sim_cfg.imu_rate
    odom_every = int(sim_cfg.imu_rate / sim_cfg.odom_rate)
    n_steps = int(sim_cfg.duration * sim_cfg.imu_rate)

    for step in range(n_steps):
        t = step * dt

        # Set commanded velocity
        v_cmd, omega_cmd = compute_velocity_command(t)
        robot.set_velocity(v_cmd, omega_cmd)

        # Ground-truth robot integration
        robot.step(dt)

        # Sample IMU (runs at full 1 kHz)
        imu_reading = imu.read(robot, dt)

        # EKF predict (runs at full 1 kHz)
        ekf.predict(imu_reading, dt)

        # EKF update from odometry (runs at odom_rate)
        if use_odometry and (step % odom_every == 0):
            odom_reading = odom.read(robot)
            ekf.update_odometry(odom_reading)

        # Record at 50 Hz to keep memory usage reasonable
        if step % odom_every == 0:
            est = ekf.state
            result.times.append(t)
            result.true_x.append(robot.x)
            result.true_y.append(robot.y)
            result.est_x.append(est[0])
            result.est_y.append(est[1])

            pos_err = np.hypot(robot.x - est[0], robot.y - est[1])
            vel_err = abs(robot.v - est[3])

            result.pos_error.append(pos_err)
            result.vel_error.append(vel_err)

    return result


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

def run_benchmark(sim_cfg: SimConfig) -> None:
    """
    @brief Benchmark EKF computational performance.

    @details
    Measures average execution time per predict and update step.
    """
    print("\n--- Performance Benchmark ---")

    rng = np.random.default_rng(sim_cfg.seed)

    robot = Robot()
    robot.set_velocity(1.0, 0.3)

    imu = IMUSensor(rng=rng)
    odom = WheelOdometrySensor(rng=rng)
    ekf = EKF()

    dt = 1.0 / sim_cfg.imu_rate
    n = int(sim_cfg.imu_rate * 10) # 10 seconds worth of steps

    # Pre-generate readings so measurement is pure EKF time
    imu_readings = [imu.read(robot, dt) for _ in range(n)]
    odom_readings = [odom.read(robot) for _ in range(n // 20)]

    # Benchmark predict
    t0 = time.perf_counter()
    for reading in imu_readings:
        ekf.predict(reading, dt)
    t1 = time.perf_counter()

    predict_total = t1 - t0
    predict_us = predict_total / n * 1e6

    # Benchmark update
    t0 = time.perf_counter()
    for reading in odom_readings:
        ekf.update_odometry(reading)
    t1 = time.perf_counter()

    update_total = t1 - t0
    update_us = update_total / len(odom_readings) * 1e6

    print(f"  predict() : {predict_us:.2f} µs/call  "
          f"({n} calls in {predict_total*1000:.1f} ms)")
    print(f"  update()  : {update_us:.2f} µs/call  "
          f"({len(odom_readings)} calls in {update_total*1000:.1f} ms)")
    print(f"  Budget at 1 kHz: 1000 µs -- "
          f"predict uses {predict_us/1000*100:.1f}% of that budget")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scenario(
    results: Sequence[SimResult],
    title: str,
    output_path: str,
) -> None:
    """
    @brief Plot trajectory and error metrics for simulation results.

    @param results List of simulation results
    @param title Plot title
    @param output_path Output file path for saved figure
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_traj = fig.add_subplot(gs[:, 0]) # trajectory (tall)
    ax_pos = fig.add_subplot(gs[0, 1])  # position error over time
    ax_vel = fig.add_subplot(gs[1, 1])  # velocity error over time

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Ground truth
    ax_traj.plot(
        results[0].true_x, results[0].true_y,
        "k--", linewidth=1.5, label="Ground truth", zorder=10,
    )

    for i, r in enumerate(results):
        c = colours[i % len(colours)]
        ax_traj.plot(r.est_x, r.est_y, color=c, linewidth=1.0, label=r.label)
        ax_pos.plot(r.times, r.pos_error, color=c, linewidth=1.0, label=r.label)
        ax_vel.plot(r.times, r.vel_error, color=c, linewidth=1.0, label=r.label)

    ax_traj.set_title("XY Trajectory")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.legend(fontsize=8)
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)

    ax_pos.set_title("Position Error")
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Position error (m)")
    ax_pos.legend(fontsize=8)
    ax_pos.grid(True, alpha=0.3)

    ax_vel.set_title("Velocity Error")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity error (m/s)")
    ax_vel.legend(fontsize=8)
    ax_vel.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def print_summary(results: Sequence[SimResult]) -> None:
    print(f"\n{'Label':<45} {'Mean pos err (m)':>18} {'Max pos err (m)':>17}")
    print("-" * 82)
    for r in results:
        mean_e = np.mean(r.pos_error)
        max_e  = np.max(r.pos_error)
        print(f"  {r.label:<43} {mean_e:>17.3f}  {max_e:>16.3f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Robot sensor fusion demo")
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run only the performance benchmark",
    )
    parser.add_argument(
        "--scenario", choices=["cheap", "fusion", "all"], default="all",
        help="Which scenario to run (default: all)",
    )
    args = parser.parse_args(argv)

    cfg = SimConfig()

    run_benchmark(cfg)

    if args.benchmark:
        return

    print("Running simulations...")

    # ------------------------------------------------------------------
    # Scenario 1: high-quality vs cheap IMU (both with odometry)
    # ------------------------------------------------------------------
    if args.scenario in ("cheap", "all"):
        print("\nScenario 1: High-quality IMU vs cheap IMU")
        r_hq = run_simulation(IMU_HIGH_QUALITY, cfg, use_odometry=True)
        r_cheap = run_simulation(IMU_CHEAP, cfg, use_odometry=True)
        results_1 = [r_hq, r_cheap]
        print_summary(results_1)
        plot_scenario(
            results_1,
            title="Scenario 1: EKF with high-quality IMU vs cheap IMU",
            output_path="plots/scenario1_imu_quality.png",
        )

    # ------------------------------------------------------------------
    # Scenario 2: odometry fusion vs IMU-only (using cheap IMU)
    # ------------------------------------------------------------------
    if args.scenario in ("fusion", "all"):
        print("\nScenario 2: Full fusion vs IMU-only (cheap IMU)")
        r_fused = run_simulation(IMU_CHEAP, cfg, use_odometry=True)
        r_imu_only = run_simulation(IMU_CHEAP, cfg, use_odometry=False)
        results_2 = [r_fused, r_imu_only]
        print_summary(results_2)
        plot_scenario(
            results_2,
            title="Scenario 2: Full fusion (IMU + odometry) vs IMU-only dead-reckoning",
            output_path="plots/scenario2_fusion_vs_imuonly.png",
        )

    print("Done.")


if __name__ == "__main__":
    main()
