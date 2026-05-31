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

from state import StateIdx, STATE_DIM


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
        self._state = np.zeros(STATE_DIM, dtype=np.float64)
        self._state[StateIdx.PX] = x
        self._state[StateIdx.PY] = y
        self._state[StateIdx.THETA] = theta
        self._state[StateIdx.V] = 0.0
        self._state[StateIdx.OMEGA] = 0.0

    # ---------------------------------------------------------------------
    # State accessors
    # ---------------------------------------------------------------------

    @property
    def x(self) -> float:
        """@brief Current x position in metres."""
        return float(self._state[StateIdx.PX])

    @property
    def y(self) -> float:
        """@brief Current y position in metres."""
        return float(self._state[StateIdx.PY])

    @property
    def theta(self) -> float:
        """@brief Current heading in radians."""
        return float(self._state[StateIdx.THETA])

    @property
    def v(self) -> float:
        """@brief Current linear velocity in m/s."""
        return float(self._state[StateIdx.V])

    @property
    def omega(self) -> float:
        """@brief Current angular velocity in rad/s."""
        return float(self._state[StateIdx.OMEGA])

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
        self._state[StateIdx.V] = v
        self._state[StateIdx.OMEGA] = omega

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
            # Degenerate case: straight-line motion.
            #
            # As omega approaches zero, the arc radius r = v / omega diverges.
            # Falls back to direct integration to avoid division by zero.
            x += v * dt * np.cos(theta)
            y += v * dt * np.sin(theta)
        else:
            # Exact arc integration for the unicycle model under constant linear and angular velocity.
            #
            # The robot moves along a circular trajectory with radius:
            #   r = v / omega
            #
            # The instantaneous centre of curvature is fixed in the robot frame at a distance r
            # perpendicular to the current heading direction. The side (left or right) depends on
            # the sign of omega.
            #
            # The motion can be interpreted as a rotation about this centre of curvature, leading to
            # an exact closed-form update of the position over the interval dt:
            #
            #   x_new = x + r * (sin(theta + omega * dt) - sin(theta))
            #   y_new = y + r * (cos(theta) - cos(theta + omega * dt))
            #
            # This solution is exact under the assumption that linear and angular velocity remain
            # constant over dt, and it avoids numerical error associated with Euler integration.
            r = v / omega # radius of curvature of the arc
            x += r * (np.sin(theta + omega * dt) - np.sin(theta))
            y += r * (np.cos(theta) - np.cos(theta + omega * dt))

        theta += omega * dt

        # Normalise angle to [-pi, pi)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        self._state[StateIdx.PX] = x
        self._state[StateIdx.PY] = y
        self._state[StateIdx.THETA] = theta