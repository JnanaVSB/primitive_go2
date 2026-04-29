"""Hardcoded joint angle primitives for the Go2 quadruped.

Each function returns a 12-dim numpy array of joint angles in JOINT_NAMES
order: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
        RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf].

These values were computed by running known-good foot positions through
Go2Kinematics.policy_to_joints(). They are frozen here so the LLM can
call them by name without needing to know foot coordinates or IK.

Joint angle conventions (from go2.xml):
    hip (abduction): 0 (fixed for these tasks)
    thigh:           positive = rotates leg forward/down
    calf:            negative = knee folded
"""

import numpy as np


def get_stand_pose() -> np.ndarray:
    """Standing pose — all four legs extended, body level at ~0.27m.

    All legs identical. This matches the home/initial pose.
    """
    return np.array([
        0.0,  0.884337, -1.768673,  # FR: hip, thigh, calf
        0.0,  0.884337, -1.768673,  # FL
        0.0,  0.884337, -1.768673,  # RR
        0.0,  0.884337, -1.768673,  # RL
    ])


def get_lay_pose() -> np.ndarray:
    """Laying pose — all four legs tucked, body flat on ground.

    All legs identical. Thighs rotated far forward, calves fully folded.
    """
    return np.array([
        0.0,  1.350714, -2.701428,  # FR
        0.0,  1.350714, -2.701428,  # FL
        0.0,  1.350714, -2.701428,  # RR
        0.0,  1.350714, -2.701428,  # RL
    ])


def get_sit_pose() -> np.ndarray:
    """Sitting pose — front legs extended, rear legs tucked under body.

    Front legs hold the same angles as standing. Rear legs fold tight
    with thighs rotated far forward and calves bent.
    """
    return np.array([
        0.0,  0.884337, -1.768673,  # FR: same as stand
        0.0,  0.884337, -1.768673,  # FL: same as stand
        0.0,  2.604590, -2.297253,  # RR: tucked
        0.0,  2.604590, -2.297253,  # RL: tucked
    ])


def get_walk_planner(
    stride_length: float = 0.10,
    swing_height: float = 0.08,
    body_height: float = 0.27,
    cycle_period: float = 0.4,
) -> dict:
    """Create a walk gait planner with the given parameters.

    Returns a dict containing the planner and cycle_period so the LLM
    can use it in a walk loop. The planner uses a Bezier trot gait
    based on the MIT Cheetah trajectory planner.

    Usage:
        walk = get_walk_planner(stride_length=0.10, swing_height=0.08)
        planner = walk['planner']
        period = walk['cycle_period']

        for step in range(n_steps):
            phi = (step * dt / period) % 1.0
            joints = walk_step(planner, phi)
            robot.set_joints(joints)
            robot.step(dt)

    Args:
        stride_length: total forward foot travel per cycle in meters (default 0.10).
        swing_height:  peak foot lift during swing in meters (default 0.08).
        body_height:   nominal body height in meters (default 0.27).
        cycle_period:  seconds per full gait cycle (default 0.4).

    Returns:
        dict with 'planner' (BezierGaitPlanner) and 'cycle_period' (float).
    """
    from world.walk_gait import make_walk_planner
    return {
        'planner': make_walk_planner(
            stride_length=stride_length,
            swing_height=swing_height,
            body_height=body_height,
        ),
        'cycle_period': cycle_period,
    }