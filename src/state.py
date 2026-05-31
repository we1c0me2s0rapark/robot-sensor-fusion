"""
@file state.py
@brief Shared state definition for robot and EKF

@details
Defines the structure and indexing of the system state vector used across
robot simulation, sensor models, and estimation (EKF).

State vector:
    [px, py, theta, v, omega]

where:
    px, py : position in metres
    theta  : heading in radians
    v      : linear velocity in m/s
    omega  : angular velocity in rad/s
"""

from enum import IntEnum


class StateIdx(IntEnum):
    """
    @brief Index mapping for the system state vector
    """
    PX = 0
    PY = 1
    THETA = 2
    V = 3
    OMEGA = 4


# State dimension (used for vector/matrix allocation)
STATE_DIM = 5


# Optional: human-readable labels (useful for debugging / plotting)
STATE_LABELS = ["px", "py", "theta", "v", "omega"]