"""
@file test_sensor_fusion.py
@brief Unit and integration tests for the robot sensor fusion stack.

@details
This module contains unit tests for:
- Robot kinematics model
- IMU sensor simulation
- Wheel odometry sensor simulation
- Extended Kalman Filter (EKF)

It also includes integration and performance tests to validate:
- Numerical stability
- Sensor noise behaviour
- Sensor fusion accuracy improvements
- Real-time execution constraints

Run tests using:
    pytest tests/ -v
"""

from __future__ import annotations
import sys
import os
import math
import time

import numpy as np
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.robot import Robot
from src.sensors import IMUSensor, WheelOdometrySensor
from src.ekf import EKF


# ---------------------------------------------------------------------------
# Robot model tests
# ---------------------------------------------------------------------------

class TestRobot:
    """Unit tests for the Robot kinematic model."""

    def test_initial_state_is_zero(self):
        """Robot should initialise at the origin with zero velocity."""
        r = Robot()
        assert r.x == pytest.approx(0.0)
        assert r.y == pytest.approx(0.0)
        assert r.theta == pytest.approx(0.0)
        assert r.v == pytest.approx(0.0)
        assert r.omega == pytest.approx(0.0)

    def test_straight_line_motion(self):
        """Straight motion should produce linear displacement in x."""
        r = Robot()
        r.set_velocity(1.0, 0.0)
        r.step(1.0)
        assert r.x == pytest.approx(1.0, abs=1e-9)
        assert r.y == pytest.approx(0.0, abs=1e-9)

    def test_circular_motion_returns_to_origin(self):
        """Full circular motion should return approximately to origin."""
        r = Robot()
        r.set_velocity(1.0, math.pi)
        dt = 0.001
        for _ in range(2000):   # 2 seconds == one full circle
            r.step(dt)
        assert r.x == pytest.approx(0.0, abs=0.05)
        assert r.y == pytest.approx(0.0, abs=0.05)

    def test_heading_stays_normalised(self):
        """Heading angle must remain within (-pi, pi]."""
        r = Robot()
        r.set_velocity(0.0, 4.0)
        for _ in range(10000):
            r.step(0.001)
        assert -math.pi < r.theta <= math.pi

    def test_state_copy(self):
        """State accessor must return a copy, not a reference."""
        r = Robot(x=1.0, y=2.0)
        s = r.state
        s[0] = 99.0
        assert r.x == pytest.approx(1.0)    # mutation should not affect robot


# ---------------------------------------------------------------------------
# IMU sensor tests
# ---------------------------------------------------------------------------

class TestIMUSensor:
    """Unit tests for IMU noise and bias behaviour."""

    def test_noiseless_reading_near_true_value(self):
        """With zero noise and drift, IMU output should match truth."""
        r = Robot()
        r.set_velocity(1.0, 0.0)
        imu = IMUSensor(
            accel_noise_std=0.0,
            gyro_noise_std=0.0,
            accel_bias_drift=0.0,
            gyro_bias_drift=0.0,
        )
        reading = imu.read(r, dt=0.001)
        assert reading["omega"] == pytest.approx(0.0, abs=1e-9)

    def test_noise_is_stochastic(self):
        """IMU noise should exhibit non-zero variance."""
        r = Robot()
        r.set_velocity(1.0, 0.3)
        imu = IMUSensor(accel_noise_std=0.1, gyro_noise_std=0.05)
        readings = [imu.read(r, dt=0.001)["omega"] for _ in range(500)]
        assert np.std(readings) > 0.001   # some variance expected

    def test_drift_accumulates(self):
        """Bias drift should increase over time."""
        r = Robot()
        r.set_velocity(1.0, 0.0)
        imu = IMUSensor(
            accel_noise_std=0.0,
            gyro_noise_std=0.0,
            accel_bias_drift=0.0,
            gyro_bias_drift=0.1,    # large drift
        )
        bias_early = abs(imu.read(r, dt=0.001)["omega"])
        for _ in range(5000):
            imu.read(r, dt=0.001)
        bias_late = abs(imu.read(r, dt=0.001)["omega"])
        assert bias_late > bias_early

    def test_reset_clears_bias(self):
        """Reset should clear accumulated bias."""
        r = Robot()
        r.set_velocity(1.0, 0.0)
        imu = IMUSensor(gyro_noise_std=0.0, gyro_bias_drift=0.5)
        for _ in range(1000):
            imu.read(r, dt=0.001)
        imu.reset()
        reading = imu.read(r, dt=0.001)
        # After reset, bias is zero again so reading should be near truth
        assert abs(reading["omega"]) < 0.5


# ---------------------------------------------------------------------------
# Wheel odometry tests
# ---------------------------------------------------------------------------

class TestWheelOdometrySensor:
    """Unit tests for wheel odometry behaviour."""

    def test_noiseless_straight(self):
        r = Robot()
        r.set_velocity(1.0, 0.0)
        odom = WheelOdometrySensor(noise_std=0.0, slip_probability=0.0)
        reading = odom.read(r)
        assert reading["v"]     == pytest.approx(1.0, abs=1e-9)
        assert reading["omega"] == pytest.approx(0.0, abs=1e-9)

    def test_noiseless_turning(self):
        r = Robot()
        r.set_velocity(1.0, 0.5)
        odom = WheelOdometrySensor(wheel_base=0.5, noise_std=0.0, slip_probability=0.0)
        reading = odom.read(r)
        assert reading["v"]     == pytest.approx(1.0, abs=1e-9)
        assert reading["omega"] == pytest.approx(0.5, abs=1e-9)

    def test_slip_event_changes_reading(self):
        """Wheel slip should introduce deviation in velocity estimate."""
        r = Robot()
        r.set_velocity(2.0, 0.0)
        # With slip_probability=1.0, every reading will have a slip event
        odom = WheelOdometrySensor(noise_std=0.0, slip_probability=1.0)
        readings = [odom.read(r) for _ in range(100)]
        # At least some readings should deviate from true v=2.0
        velocities = [rd["v"] for rd in readings]
        assert min(velocities) < 1.9


# ---------------------------------------------------------------------------
# EKF tests
# ---------------------------------------------------------------------------

class TestEKF:
    """Unit and integration tests for the Extended Kalman Filter."""

    def test_predict_does_not_raise(self):
        """Predict step should execute without error."""
        ekf = EKF()
        imu_reading = {"accel": np.array([0.0, 0.0]), "omega": 0.0}
        ekf.predict(imu_reading, dt=0.001)

    def test_update_does_not_raise(self):
        """Update step should execute without error."""
        ekf = EKF()
        ekf.update_odometry({"v": 1.0, "omega": 0.0})

    def test_stationary_robot_stays_near_origin(self):
        """Zero motion should not produce significant drift."""
        ekf = EKF()
        imu_reading = {"accel": np.zeros(2), "omega": 0.0}
        for _ in range(1000):
            ekf.predict(imu_reading, dt=0.001)
            ekf.update_odometry({"v": 0.0, "omega": 0.0})
        x, y = ekf.position
        assert abs(x) < 0.05
        assert abs(y) < 0.05

    def test_covariance_is_positive_definite(self):
        """Covariance matrix should remain positive definite."""
        ekf = EKF()
        imu_reading = {"accel": np.array([0.1, 0.0]), "omega": 0.3}
        for _ in range(100):
            ekf.predict(imu_reading, dt=0.001)
        eigenvalues = np.linalg.eigvalsh(ekf.covariance)
        assert np.all(eigenvalues > 0), "Covariance should remain positive definite"

    def test_odometry_update_reduces_uncertainty(self):
        """Odometry update should reduce velocity variance."""
        ekf = EKF(init_cov_diag=(1.0, 1.0, 0.1, 5.0, 0.5))
        cov_before = ekf.covariance[3, 3]   # velocity variance
        ekf.update_odometry({"v": 1.0, "omega": 0.0})
        cov_after = ekf.covariance[3, 3]
        assert cov_after < cov_before

    def test_fusion_more_accurate_than_imu_only(self):
        """Sensor fusion should outperform IMU-only estimation."""
        from src.sensors import IMUSensor, WheelOdometrySensor
        from src.robot import Robot

        robot_a = Robot()
        robot_b = Robot()
        imu_a = IMUSensor(gyro_noise_std=0.05, gyro_bias_drift=0.01, rng=np.random.default_rng(1))
        imu_b = IMUSensor(gyro_noise_std=0.05, gyro_bias_drift=0.01, rng=np.random.default_rng(1))
        odom = WheelOdometrySensor(noise_std=0.02, rng=np.random.default_rng(2))

        ekf_fused = EKF()
        ekf_imu_only = EKF()

        dt = 0.001
        for step in range(5000):
            robot_a.set_velocity(1.0, 0.3)
            robot_b.set_velocity(1.0, 0.3)
            robot_a.step(dt)
            robot_b.step(dt)

            r_a = imu_a.read(robot_a, dt)
            r_b = imu_b.read(robot_b, dt)

            ekf_fused.predict(r_a, dt)
            ekf_imu_only.predict(r_b, dt)

            if step % 20 == 0:
                ekf_fused.update_odometry(odom.read(robot_a))

        err_fused = math.hypot(robot_a.x - ekf_fused.position[0],
                               robot_a.y - ekf_fused.position[1])
        err_imu_only = math.hypot(robot_b.x - ekf_imu_only.position[0],
                                  robot_b.y - ekf_imu_only.position[1])
        assert err_fused < err_imu_only


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------

class TestPerformance:
    """Real-time performance constraints for EKF execution."""

    def test_predict_faster_than_100us(self):
        """Predict step must be fast enough for 1 kHz execution."""
        ekf = EKF()
        imu_reading = {"accel": np.array([0.1, 0.0]), "omega": 0.3}
        n = 10_000
        t0 = time.perf_counter()
        for _ in range(n):
            ekf.predict(imu_reading, dt=0.001)
        elapsed_us = (time.perf_counter() - t0) / n * 1e6
        assert elapsed_us < 100.0, (
            f"predict() took {elapsed_us:.1f} µs -- must be < 100 µs for 1 kHz operation"
        )

    def test_update_faster_than_200us(self):
        """Update step must be fast enough for real-time use."""
        ekf = EKF()
        odom_reading = {"v": 1.0, "omega": 0.3}
        n = 10_000
        t0 = time.perf_counter()
        for _ in range(n):
            ekf.update_odometry(odom_reading)
        elapsed_us = (time.perf_counter() - t0) / n * 1e6
        assert elapsed_us < 200.0, (
            f"update_odometry() took {elapsed_us:.1f} µs -- must be < 200 µs"
        )
