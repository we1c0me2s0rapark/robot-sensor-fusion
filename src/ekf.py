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

from state import StateIdx, STATE_DIM


# ---------------------------------------------------------------------
# State and measurement dimensions
# ---------------------------------------------------------------------

_N = STATE_DIM  # state dimension: [px, py, theta, v, omega]
_M = 2          # measurement dimension: [v, omega]


class EKF:
    """
    @brief Extended Kalman Filter for robot state estimation.

    @details
    Implements a nonlinear motion model (unicycle kinematics) combined with
    linearised measurement updates for wheel odometry.
    """

    def __init__(
        self,
        # standard deviation of process noise for [px, py, theta, v, omega]
        process_noise_std: tuple[float, float, float, float, float] = (
            0.01, 0.01, 0.001, 0.1, 0.01
        ),
        # standard deviation of odometry measurement noise for [v, omega]
        odom_noise_std: tuple[float, float] = (
            0.05, 0.01
        ),
        # initial state vector [px, py, theta, v, omega]
        init_state: np.ndarray | None = None,
        # initial diagonal covariance values for [px, py, theta, v, omega]
        init_cov_diag: tuple[float, float, float, float, float] = (
            1.0, 1.0, 0.1, 1.0, 0.1
        ),
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

        # Initial state defaults to zero position, heading, and velocity if not provided
        self._x = (
            np.array(init_state, dtype=np.float64)
            if init_state is not None
            else np.zeros(_N, dtype=np.float64)
        )

        # Initial state covariance matrix (diagonal)
        # Represents the initial uncertainty in the state estimate
        # Squared values correspond to variances of each state component
        self._P = np.diag(np.array(init_cov_diag, dtype=np.float64))

        # -----------------------------------------------------------------
        # Noise covariance matrices
        # -----------------------------------------------------------------
        
        # Process noise covariance `Q`
        # Standard deviations are converted to variances for each state component
        # Q models uncertainty in the motion (process) model
        q = np.array(process_noise_std, dtype=np.float64) ** 2
        self._Q_base = np.diag(q)   # scaled by dt in predict()

        # Measurement noise covariance `R`
        # Standard deviations are converted to variances for the odometry measurements
        # R models uncertainty in the sensor measurements
        r = np.array(odom_noise_std, dtype=np.float64) ** 2
        self._R = np.diag(r)

        # -----------------------------------------------------------------
        # Measurement model
        # -----------------------------------------------------------------

        # Wheel odometry provides noisy measurements of:
        # [linear velocity v, angular velocity omega]
        #
        # These correspond to state indices [3, 4] in:
        # x = [px, py, theta, v, omega]
        #
        # Linear measurement model:
        # z = Hx + noise
        #
        # H selects v and omega directly from the state vector
        self._H = np.zeros((_M, _N), dtype=np.float64)
        self._H[0, StateIdx.V] = 1.0
        self._H[1, StateIdx.OMEGA] = 1.0

        # -----------------------------------------------------------------
        # Pre-allocated matrices for EKF performance
        # -----------------------------------------------------------------

        # State transition Jacobian `F`
        # Linearises the nonlinear motion model around the current estimate
        # F = df(x) / dx
        #
        # Used in covariance propagation:
        # P = F P F^T + Q (process noise)
        #
        # Recomputed each predict step from current state and IMU input
        # Initialised to identity - overwritten at the start of each predict step
        self._F = np.eye(_N, dtype=np.float64)

        # Innovation covariance `S`
        # Uncertainty of the innovation (measurement residual)
        # S = H P H^T + R
        #
        # Combines projected state uncertainty (H P H^T) with measurement noise (R)
        #
        # S defines how uncertain the residual is in measurement space
        #
        # Dimension is measurement space (2x2 for [v, omega])
        self._S = np.zeros((_M, _M), dtype=np.float64)

        # Kalman gain `K`
        # Determines how strongly the state is corrected by the measurement
        # K ≈ prediction uncertainty / total uncertainty
        # K = P H^T S^-1
        #
        # numerator P H^T: uncertainty of the state prediction
        # denominator S: total uncertainty in measurement space (prediction + noise)
        # 
        # Derived from minimising posterior covariance under Gaussian noise assumptions
        # Equivalent to a weighted least-squares update
        #
        # S controls measurement confidence:
        # Large S or small P → small K → trust prediction more
        # Small S or large P → large K → trust measurement more
        self._K = np.zeros((_N, _M), dtype=np.float64)

        # Identity matrix `I`
        # Precomputed to avoid repeated allocation in the update step
        self._I = np.eye(_N, dtype=np.float64)

        # State correction operator (I - K H)
        # K H: how the measurement update influences each state component
        # I - K H: represents the portion of prior uncertainty remaining after measurement update
        #
        # Used in the Joseph form covariance update:
        # P = (I - K H) P (I - K H)^T + K R K^T
        #
        # Joseph form preserves symmetry and positive definiteness of P
        # under finite precision arithmetic
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
        return float(self._x[StateIdx.PX]), float(self._x[StateIdx.PY])

    @property
    def heading(self) -> float:
        """@brief Return estimated heading in radians."""
        return float(self._x[StateIdx.THETA])

    @property
    def velocity(self) -> float:
        """@brief Return estimated linear velocity in m/s."""
        return float(self._x[StateIdx.V])

    @property
    def angular_velocity(self) -> float:
        """@brief Return estimated angular velocity in rad/s."""
        return float(self._x[StateIdx.OMEGA])

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

        # Heading propagation using angular velocity from gyroscope
        # theta_{k+1} = theta_k + omega * dt assuming constant angular rate over dt
        #
        # Wrapped to [-pi, pi) to prevent angle discontinuities
        # and ensure consistent representation across all computations
        new_theta = theta + omega_meas * dt
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi

        # IMU acceleration is provided in the world frame (x, y)
        # The EKF velocity state is defined along the robot heading direction
        #
        # Project acceleration onto the robot forward axis using dot product:
        # Heading unit vector: [cos(theta), sin(theta)]
        # a_longitudinal = ax * cos(theta) + ay * sin(theta)
        #
        # This extracts the component of acceleration that affects forward speed
        ax, ay = imu_reading["accel"]
        a_longitudinal = ax * np.cos(theta) + ay * np.sin(theta)

        # Linear velocity update using projected longitudinal acceleration
        # Assumes constant acceleration over the timestep (first-order Euler integration)
        new_v = v + a_longitudinal * dt

        # Angular velocity is directly taken from the gyroscope measurement
        # and used as the instantaneous angular rate for motion propagation
        new_omega = omega_meas

        self._x[StateIdx.PX] = new_px
        self._x[StateIdx.PY] = new_py
        self._x[StateIdx.THETA] = new_theta
        self._x[StateIdx.V] = new_v
        self._x[StateIdx.OMEGA] = new_omega

        # -----------------------------------------------------------------
        # Jacobian computation
        # -----------------------------------------------------------------
        # Compute F = df/dx evaluated at the current (pre-update) state
        #
        # F is stored in a preallocated buffer (self._F) to avoid allocations
        # It is reset to identity because the motion model is identity plus
        # nonlinear coupling terms between states
        #
        # This avoids constructing a full new matrix each predict step
        F = self._F
        F[:] = self._I  # reset to identity before filling non-trivial Jacobian entries

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # If omega ≈ 0, use straight-line motion Jacobian; otherwise, use arc motion Jacobian
        if abs(omega_meas) < 1e-9:
            # d(px)/d(theta) and d(py)/d(theta): position sensitivity to heading in straight-line motion
            F[StateIdx.PX, StateIdx.THETA] = -v * dt * sin_t
            F[StateIdx.PY, StateIdx.THETA] =  v * dt * cos_t
            # d(px)/d(v) and d(py)/d(v): position sensitivity to linear velocity
            F[StateIdx.PX, StateIdx.V] = dt * cos_t
            F[StateIdx.PY, StateIdx.V] = dt * sin_t
        else:
            r = v / omega_meas
            cos_t1 = np.cos(theta + omega_meas * dt)
            sin_t1 = np.sin(theta + omega_meas * dt)

            # d(px)/d(theta) and d(py)/d(theta): position sensitivity to heading in arc motion
            F[StateIdx.PX, StateIdx.THETA] = r * (cos_t1 - cos_t)
            F[StateIdx.PY, StateIdx.THETA] = r * (sin_t1 - sin_t)
            # d(px)/d(omega) and d(py)/d(omega): position sensitivity to turning rate
            F[StateIdx.PX, StateIdx.OMEGA] = (sin_t1 - sin_t) / omega_meas
            F[StateIdx.PY, StateIdx.OMEGA] = (cos_t - cos_t1) / omega_meas

        # d(v)/d(theta): effect of heading on projected longitudinal acceleration
        F[StateIdx.V, StateIdx.THETA] = (-ax * np.sin(theta) + ay * np.cos(theta)) * dt

        # -----------------------------------------------------------------
        # Covariance propagation using linearised dynamics
        # -----------------------------------------------------------------
        # Propagate state uncertainty through the motion model
        #
        # F P F^T:
        #   Transforms the previous covariance through the linearised dynamics
        #   - F maps how each state component influences others over dt
        #   - P is the prior uncertainty
        #   - F^T preserves symmetry and correct tensor transformation
        #
        # Q:
        #   Process noise covariance representing uncertainty introduced by
        #   unmodelled dynamics (e.g. acceleration noise, slip, IMU errors)
        #
        # Noise is scaled by dt because uncertainty accumulates over time
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

        # Innovation (residual)
        # y measures the mismatch between the actual measurement and the predicted measurement
        #
        # z: actual odometry measurement [v, omega]
        # Hx: predicted measurement obtained from current state estimate [v_estimated, omega_estimated]
        #
        # Interpretation:
        # - y[0]: velocity error (measured v minus estimated v)
        # - y[1]: angular velocity error (measured omega minus estimated omega)
        #
        # This residual is the key signal used to correct the state estimate
        # in proportion to its uncertainty via the Kalman gain
        y = z - self._H @ self._x

        # Innovation covariance S
        # S measures the total uncertainty of the innovation (residual) y = z - Hx
        #
        # S = H P H^T + R
        #
        # where:
        # H P H^T: projects state uncertainty P into measurement space
        #         (uncertainty in predicted measurement due to uncertainty in x)
        #
        # R: measurement noise covariance
        #   (sensor uncertainty, independent of state)
        #
        # Interpretation:
        # S represents how uncertain the residual is expected to be before seeing the measurement
        #
        # This is used to weight the Kalman gain:
        # - large S → low trust in measurement update
        # - small S → high trust in measurement update
        S = self._S
        np.copyto(S, self._H @ self._P @ self._H.T + self._R)
        
        # -----------------------------------------------------------------
        # Kalman gain computation
        # -----------------------------------------------------------------

        # Explicit inverse of the 2x2 innovation covariance matrix S
        #
        # For:
        #
        #     [a b]
        # S = [c d]
        #
        # det(S) = ad - bc
        #
        #             1        [ d -b]
        # S^-1 = ----------- * [-c  a]
        #           det(S)
        #
        # Using the closed-form inverse is faster than np.linalg.inv()
        # for the fixed 2x2 measurement dimension [v, omega]
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        S_inv = np.array(
            [[ S[1, 1], -S[0, 1]],
             [-S[1, 0],  S[0, 0]]],
            dtype=np.float64,
        ) / det

        # Kalman gain:
        # K = P H^T S^-1
        #
        # Maps measurement residuals into state corrections
        #
        # Large K:
        #   - prediction is uncertain (large P)
        #   - measurement is reliable (small S)
        #
        # Small K:
        #   - prediction is confident (small P)
        #   - measurement is noisy (large S)
        K = self._K
        np.copyto(K, self._P @ self._H.T @ S_inv)

        # -----------------------------------------------------------------
        # State correction
        # -----------------------------------------------------------------

        # Apply measurement correction:
        # x = x + K y
        #
        # y is the innovation (measurement residual):
        # y = z - Hx
        #
        # The correction magnitude is determined by K
        self._x += K @ y

        # Normalise heading to [-pi, pi) to maintain a unique angle representation
        self._x[StateIdx.THETA] = (self._x[StateIdx.THETA] + np.pi) % (2 * np.pi) - np.pi

        # -----------------------------------------------------------------
        # Covariance correction
        # -----------------------------------------------------------------

        # State correction operator:
        # (I - K H)
        #
        # Represents the portion of prior uncertainty remaining
        # after incorporating the measurement
        IKH = self._IKH
        np.copyto(IKH, self._I - K @ self._H)

        # Joseph-form covariance update:
        #
        # P = (I - K H) P (I - K H)^T + K R K^T
        #
        # First term:
        #   Remaining uncertainty after the measurement update
        #
        # Second term:
        #   Measurement noise mapped into state space
        #
        # Joseph form is numerically more stable than:
        #   P = (I - K H) P
        #
        # and better preserves symmetry and positive definiteness
        # under finite-precision arithmetic
        self._P = IKH @ self._P @ IKH.T + K @ self._R @ K.T