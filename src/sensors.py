"""
@file sensors.py
@brief Virtual sensor models for IMU and wheel odometry.

@details
This module defines simulated sensor models used for robot state estimation
experiments. Each sensor reads the ground-truth Robot state and applies
noise, bias drift, and occasional disturbances.

The models are designed for Extended Kalman Filter (EKF) evaluation and
robustness testing under realistic measurement imperfections.

Sensor models included:
- IMUSensor: accelerometer and gyroscope with noise and bias drift
- WheelOdometrySensor: differential drive wheel encoder model
"""

from __future__ import annotations
import numpy as np
from src.robot import Robot


class IMUSensor:
    """
    @brief Simulated inertial measurement unit (IMU).

    @details
    This sensor models a 6-DOF IMU with:
    - Gaussian white noise
    - Bias drift modelled as a discrete-time random walk
    - Optional temporal correlation via bias accumulation

    Outputs:
    - Linear acceleration (ax, ay) in m/s^2
    - Angular velocity (omega) in rad/s
    """

    def __init__(
        self,
        accel_noise_std: float = 0.01,
        gyro_noise_std: float = 0.001,
        accel_bias_drift: float = 0.0001,
        gyro_bias_drift: float = 0.00001,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        @brief Initialise IMU sensor model.

        @param accel_noise_std Standard deviation of acceleration noise
        @param gyro_noise_std Standard deviation of gyroscope noise
        @param accel_bias_drift Standard deviation of acceleration bias drift per sqrt(dt)
        @param gyro_bias_drift Standard deviation of gyroscope bias drift per sqrt(dt)
        @param rng Optional random number generator for reproducibility
        """
        # White noise parameters
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std

        # Bias drift parameters; slowly evolving bias modeled as a random walk
        self.accel_bias_drift = accel_bias_drift
        self.gyro_bias_drift = gyro_bias_drift

        # Random number generator for noise and bias drift
        self._rng = rng or np.random.default_rng()

        # Persistent biases (start at zero, drift over time)
        self._accel_bias = np.zeros(2, dtype=np.float64) # [bias_ax, bias_ay]
        self._gyro_bias = 0.0

    def reset(self) -> None:
        """
        @brief Reset internal bias states to zero.
        """
        self._accel_bias[:] = 0.0
        self._gyro_bias = 0.0

    def read(self, robot: Robot, dt: float) -> dict[str, np.ndarray | float]:
        """
        @brief Sample IMU measurement from ground-truth robot state.

        @param robot Ground-truth robot instance
        @param dt Time step in seconds used for bias evolution

        @return Dictionary containing:
            - accel: numpy array (ax, ay) in m/s^2
            - omega: angular velocity in rad/s
        """

        # True linear acceleration in the world frame for a unicycle model
        #
        # Assumptions:
        # - Linear speed v is constant over the timestep
        # - Angular velocity omega is constant over the timestep
        # - Heading evolves as: theta_dot = omega
        #
        # Velocity in world frame:
        #   v_x = v * cos(theta)
        #   v_y = v * sin(theta)
        #
        # Acceleration is the time derivative of velocity:
        # (only theta changes with time, v is constant)
        #
        #   a_x = d/dt (v * cos(theta)) = -v * omega * sin(theta)
        #   a_y = d/dt (v * sin(theta)) =  v * omega * cos(theta)
        #
        # Geometric interpretation:
        # - Velocity is tangent to the motion
        # - Acceleration is perpendicular to velocity (90° rotation)
        # - This is purely centripetal acceleration (no change in speed)
        # - Direction corresponds to rotation of velocity vector by sign of omega
        #
        # Magnitude:
        #   |a| = v * omega
        v, omega, theta = robot.v, robot.omega, robot.theta

        true_ax = -v * omega * np.sin(theta)
        true_ay = v * omega * np.cos(theta)

        true_omega = omega

        # Update bias using Gaussian noise scaled by sqrt(dt) (random walk model)
        mean = 0.0
        sqrt_dt = np.sqrt(dt)

        # Add white noise and bias to acceleration
        accel = np.array([true_ax, true_ay], dtype=np.float64)

        # 1. Add zero-mean Gaussian white noise (uncorrelated in time)
        accel += self._rng.normal(mean, self.accel_noise_std, size=2)

        # 2. Add bias (slowly varying offset that introduces temporal correlation)
        accel_bias_std = self.accel_bias_drift * sqrt_dt
        self._accel_bias += (self._rng.normal(mean, accel_bias_std, size=2))
        accel += self._accel_bias
        
        # Add white noise and bias to angular velocity
        meas_omega = true_omega
        
        # 1. Add zero-mean Gaussian white noise (uncorrelated in time)
        meas_omega += self._rng.normal(mean, self.gyro_noise_std)
        
        # 2. Add bias (slowly varying offset that introduces temporal correlation)
        gyro_bias_std = self.gyro_bias_drift * sqrt_dt
        self._gyro_bias += self._rng.normal(mean, gyro_bias_std)
        meas_omega += self._gyro_bias

        return {"accel": accel, "omega": float(meas_omega)}


class WheelOdometrySensor:
    """
    @brief Differential drive wheel odometry sensor model

    @details
    This model simulates wheel encoder measurements for a differential drive robot.

    It includes:
    - Conversion between unicycle velocities and wheel angular velocities
    - Gaussian encoder noise
    - Optional stochastic wheel slip events
    """

    def __init__(
        self,
        wheel_base: float = 0.5,
        noise_std: float = 0.01,
        slip_probability: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        @brief Initialise wheel odometry sensor

        @param wheel_base Distance between left and right wheels (m)
        @param noise_std Standard deviation of encoder velocity noise (m/s)
        @param slip_probability Probability of a wheel slip event per update
        @param rng Random number generator for reproducibility
        """
        self.wheel_base = wheel_base
        self.noise_std = noise_std
        self.slip_probability = slip_probability
        self._rng = rng or np.random.default_rng()

    def read(self, robot: Robot) -> dict[str, float]:
        """
        @brief Generate noisy wheel odometry measurements from ground truth

        @param robot Ground-truth robot state source

        @return Dictionary containing:
            - v_left: left wheel linear velocity (m/s)
            - v_right: right wheel linear velocity (m/s)
            - v: reconstructed linear velocity estimate (m/s)
            - omega: reconstructed angular velocity estimate (rad/s)
        """

        v, omega = robot.v, robot.omega
        half_base = self.wheel_base / 2.0

        # -----------------------------------------------------------------
        # Forward kinematics: unicycle -> differential drive wheels
        #
        # v_left  = v - (omega * L/2)
        # v_right = v + (omega * L/2)
        #
        # Interpretation:
        # - omega > 0 → right wheel faster than left wheel (left turn)
        # - omega < 0 → left wheel faster than right wheel (right turn)
        # -----------------------------------------------------------------
        v_left = v - omega * half_base
        v_right = v + omega * half_base

        # Add Gaussian noise
        v_left += self._rng.normal(0.0, self.noise_std)
        v_right += self._rng.normal(0.0, self.noise_std)

        # -----------------------------------------------------------------
        # Optional wheel slip model
        #
        # With probability slip_probability:
        # - one wheel temporarily loses traction and reports zero velocity
        # - introduces strong asymmetry and odometry error
        # -----------------------------------------------------------------
        if self.slip_probability > 0.0:
            if self._rng.random() < self.slip_probability:
                if self._rng.random() < 0.5:
                    v_left = 0.0
                else:
                    v_right = 0.0

        # -----------------------------------------------------------------
        # Inverse kinematics: differential drive wheels -> unicycle
        #
        # v = (v_right + v_left) / 2
        # omega = (v_right - v_left) / L
        #
        # Interpretation:
        # - positive omega → robot turns left
        # - negative omega → robot turns right
        # -----------------------------------------------------------------
        v_est = (v_right + v_left) / 2.0
        omega_est = (v_right - v_left) / self.wheel_base

        return {
            "v_left": float(v_left),
            "v_right": float(v_right),
            "v": float(v_est),
            "omega": float(omega_est),
        }