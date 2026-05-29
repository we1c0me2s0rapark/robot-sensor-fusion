"""
@file robot.py
@brief Ground truth robot simulator using a unicycle kinematic model.

@details
This module implements a simple planar mobile robot model used as ground truth
for sensor fusion experiments.

The robot state is defined as:
    [x, y, theta, v, omega]

where:
    x      : position in metres along the x-axis
    y      : position in metres along the y-axis
    theta  : robot yaw angle in radians
    v      : linear velocity in m/s
    omega  : angular velocity in rad/s

No environmental effects such as collisions, friction variations, or slip are modelled.
"""

from __future__ import annotations
import numpy as np


class Robot:
    """
    @brief Ground truth wheeled robot using a unicycle kinematic model.

    @details
    This class maintains the true internal state of the robot.
    Sensor models sample this state and apply noise and bias to simulate measurements.

    The model applies commanded velocities directly with no actuator dynamics, latency, or saturation.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
    ) -> None:
        """
        @brief Initialise robot state.

        @param x Initial x position in metres
        @param y Initial y position in metres
        @param theta Initial heading in radians
        """
        # True state vector: [x, y, theta, v, omega]
        self._state = np.array([x, y, theta, 0.0, 0.0], dtype=np.float64)

    # ---------------------------------------------------------------------
    # State accessors
    # ---------------------------------------------------------------------

    @property
    def x(self) -> float:
        """@brief Current x position in metres."""
        return float(self._state[0])

    @property
    def y(self) -> float:
        """@brief Current y position in metres."""
        return float(self._state[1])

    @property
    def theta(self) -> float:
        """@brief Current heading in radians."""
        return float(self._state[2])

    @property
    def v(self) -> float:
        """@brief Current linear velocity in m/s."""
        return float(self._state[3])

    @property
    def omega(self) -> float:
        """@brief Current angular velocity in rad/s."""
        return float(self._state[4])

    @property
    def state(self) -> np.ndarray:
        """
        @brief Return a copy of the full robot state.

        @return Array containing [x, y, theta, v, omega]
        """
        return self._state.copy()

    # ---------------------------------------------------------------------
    # Control interface
    # ---------------------------------------------------------------------

    def set_velocity(self, v: float, omega: float) -> None:
        """
        @brief Set commanded velocity inputs.

        @param v Linear velocity in m/s
        @param omega Angular velocity in rad/s
        """
        self._state[3] = v
        self._state[4] = omega

    def step(self, dt: float) -> None:
        """
        @brief Propagate robot state forward in time.

        @param dt Time step in seconds

        @details
        The motion model follows a unicycle kinematic formulation.
        Exact arc integration is used to reduce numerical error for
        non-zero angular velocity.
        """
        x, y, theta, v, omega = self._state

        if abs(omega) < 1e-9:
            # Straight-line motion (unicycle model limit as angular velocity approaches zero)
            x += v * dt * np.cos(theta)
            y += v * dt * np.sin(theta)
        else:
            # Exact unicycle integration assuming constant linear and angular velocity over dt
            r = v / omega
            x += r * (np.sin(theta + omega * dt) - np.sin(theta))
            y += r * (np.cos(theta) - np.cos(theta + omega * dt))

        theta += omega * dt

        # Normalise angle to [-pi, pi)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        self._state[0] = x
        self._state[1] = y
        self._state[2] = theta