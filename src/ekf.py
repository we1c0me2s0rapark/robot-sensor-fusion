"""
@file ekf.py
@brief Extended Kalman Filter (EKF) for fusing IMU and wheel odometry.

@details
This module implements an Extended Kalman Filter for state estimation of a
differential drive robot.

The filter fuses:
- IMU measurements (accelerometer and gyroscope)
- Wheel odometry measurements (linear and angular velocity)

State representation:
    x = [px, py, theta, v, omega]

where:
    px, py : position in metres
    theta  : heading in radians
    v      : linear velocity in m/s
    omega  : angular velocity in rad/s

Algorithm structure:
- Prediction step runs at IMU frequency (typically 1 kHz)
- Update step runs at odometry frequency (typically 50 to 200 Hz)

Design goals:
- Pure NumPy implementation with no external dependencies
- Pre-allocated matrices to avoid runtime memory allocation
- Explicit Jacobians for performance and clarity
- Numerically stable covariance updates
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------
# State and measurement dimensions
# ---------------------------------------------------------------------

_N = 5  # state dimension: [px, py, theta, v, omega]
_M = 2  # measurement dimension: [v, omega]


class EKF:
    """
    @brief Extended Kalman Filter for robot state estimation.

    @details
    Implements a nonlinear motion model (unicycle kinematics) combined with
    linearised measurement updates for wheel odometry.
    """

    def __init__(
        self,
        process_noise_std: tuple[float, float, float, float, float] = (
            0.01, 0.01, 0.001, 0.1, 0.01
        ),
        odom_noise_std: tuple[float, float] = (0.05, 0.01),
        init_state: np.ndarray | None = None,
        init_cov_diag: tuple[float, ...] = (1.0, 1.0, 0.1, 1.0, 0.1),
    ) -> None:
        """
        @brief Initialise EKF state and covariance.

        @param process_noise_std Standard deviation of process noise
        @param odom_noise_std Standard deviation of odometry measurement noise
        @param init_state Initial state vector [px, py, theta, v, omega]
        @param init_cov_diag Initial diagonal covariance values
        """

        # -----------------------------------------------------------------
        # State vector and covariance initialisation
        # -----------------------------------------------------------------
        self._x = (
            np.array(init_state, dtype=np.float64)
            if init_state is not None
            else np.zeros(_N, dtype=np.float64)
        )

        self._P = np.diag(np.array(init_cov_diag, dtype=np.float64))

        # -----------------------------------------------------------------
        # Noise covariance matrices
        # -----------------------------------------------------------------
        q = np.array(process_noise_std, dtype=np.float64) ** 2
        self._Q_base = np.diag(q)   # scaled by dt in predict()

        r = np.array(odom_noise_std, dtype=np.float64) ** 2
        self._R = np.diag(r)

        # -----------------------------------------------------------------
        # Measurement model
        # -----------------------------------------------------------------
        # Odometry observes [v, omega] directly from state indices [3, 4]
        self._H = np.zeros((_M, _N), dtype=np.float64)
        self._H[0, 3] = 1.0
        self._H[1, 4] = 1.0

        # -----------------------------------------------------------------
        # Pre-allocated matrices for performance
        # -----------------------------------------------------------------
        self._F  = np.eye(_N, dtype=np.float64)   # state Jacobian
        self._S  = np.zeros((_M, _M), dtype=np.float64)
        self._K  = np.zeros((_N, _M), dtype=np.float64)
        self._I  = np.eye(_N, dtype=np.float64)
        self._IKH = np.zeros((_N, _N), dtype=np.float64)

    # ---------------------------------------------------------------------
    # Public accessors
    # ---------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """@brief Return a copy of the current state estimate."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """@brief Return a copy of the state covariance matrix."""
        return self._P.copy()

    @property
    def position(self) -> tuple[float, float]:
        """@brief Return estimated position (px, py)."""
        return float(self._x[0]), float(self._x[1])

    @property
    def heading(self) -> float:
        """@brief Return estimated heading in radians."""
        return float(self._x[2])

    @property
    def velocity(self) -> float:
        """@brief Return estimated linear velocity in m/s."""
        return float(self._x[3])

    @property
    def angular_velocity(self) -> float:
        """@brief Return estimated angular velocity in rad/s."""
        return float(self._x[4])

    # ---------------------------------------------------------------------
    # Prediction step
    # ---------------------------------------------------------------------

    def predict(self, imu_reading: dict, dt: float) -> None:
        """
        @brief EKF prediction step using IMU measurements.

        @param imu_reading Dictionary containing:
            - accel: np.ndarray [ax, ay]
            - omega: angular velocity (rad/s)
        @param dt Time step in seconds
        """

        px, py, theta, v, omega = self._x

        # Use IMU angular velocity as the measured omega for motion propagation
        omega_meas = imu_reading["omega"]

        # -----------------------------------------------------------------
        # Nonlinear motion model (unicycle kinematics)
        # -----------------------------------------------------------------
        if abs(omega_meas) < 1e-9:
            # Straight-line motion approximation (omega close to zero limit case)
            new_px = px + v * dt * np.cos(theta)
            new_py = py + v * dt * np.sin(theta)
        else:
            # Arc motion using closed-form unicycle integration (constant velocity assumption)
            r = v / omega_meas
            new_px = px + r * (np.sin(theta + omega_meas * dt) - np.sin(theta))
            new_py = py + r * (np.cos(theta) - np.cos(theta + omega_meas * dt))

        new_theta = theta + omega_meas * dt
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi

        # Project IMU acceleration onto robot heading to update velocity
        ax, ay = imu_reading["accel"]
        a_longitudinal = ax * np.cos(theta) + ay * np.sin(theta)

        new_v = v + a_longitudinal * dt
        new_omega = omega_meas  # trust the gyro for this step

        self._x[0] = new_px
        self._x[1] = new_py
        self._x[2] = new_theta
        self._x[3] = new_v
        self._x[4] = new_omega

        # -----------------------------------------------------------------
        # Jacobian computation
        # -----------------------------------------------------------------
        # F = df/dx evaluated at current (pre-update) state
        F = self._F
        F[:] = self._I  # reset to identity

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        if abs(omega_meas) < 1e-9:
            # Straight-line case Jacobian terms (position sensitivity to theta and velocity)
            F[0, 2] = -v * dt * sin_t
            F[1, 2] =  v * dt * cos_t
            # d(px)/d(v) and d(py)/d(v)
            F[0, 3] = dt * cos_t
            F[1, 3] = dt * sin_t
        else:
            r = v / omega_meas
            cos_t1 = np.cos(theta + omega_meas * dt)
            sin_t1 = np.sin(theta + omega_meas * dt)

            F[0, 2] = r * (cos_t1 - cos_t)
            F[1, 2] = r * (sin_t1 - sin_t)
            F[0, 3] = (sin_t1 - sin_t) / omega_meas
            F[1, 3] = (cos_t - cos_t1) / omega_meas

        # d(v)/d(theta): acceleration projection depends on heading
        F[3, 2] = (-ax * np.sin(theta) + ay * np.cos(theta)) * dt

        # -----------------------------------------------------------------
        # Covariance propagation using linearised dynamics
        # -----------------------------------------------------------------
        Q = self._Q_base * dt
        self._P = F @ self._P @ F.T + Q

    # ---------------------------------------------------------------------
    # Update step
    # ---------------------------------------------------------------------

    def update_odometry(self, odom_reading: dict) -> None:
        """
        @brief EKF update step using odometry measurements [v, omega].

        @param odom_reading Dictionary containing:
            - v: linear velocity (m/s)
            - omega: angular velocity (rad/s)
        """

        z = np.array([odom_reading["v"], odom_reading["omega"]], dtype=np.float64)

        # Innovation: y = z - H x
        y = z - self._H @ self._x

        # Innovation covariance: S = H P Hᵀ + R
        S = self._S
        np.copyto(S, self._H @ self._P @ self._H.T + self._R)

        # Compute Kalman gain using explicit 2x2 matrix inverse for efficiency
        # (faster than np.linalg.solve)
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        S_inv = np.array(
            [[ S[1, 1], -S[0, 1]],
             [-S[1, 0],  S[0, 0]]],
            dtype=np.float64,
        ) / det
        K = self._K
        np.copyto(K, self._P @ self._H.T @ S_inv)

        # State update: x = x + K y
        self._x += K @ y

        # Normalise heading
        self._x[2] = (self._x[2] + np.pi) % (2 * np.pi) - np.pi

        # Covariance update using Joseph form for numerical stability
        IKH = self._IKH
        np.copyto(IKH, self._I - K @ self._H)

        self._P = IKH @ self._P @ IKH.T + K @ self._R @ K.T