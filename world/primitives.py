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


def get_walk_phases() -> list[np.ndarray]:
    """Four-phase trot gait cycle.

    Phase 1: Pair A (FL+RR) swings forward, Pair B (FR+RL) planted.
    Phase 2: All legs planted (transition).
    Phase 3: Pair B (FR+RL) swings forward, Pair A (FL+RR) planted.
    Phase 4: All legs planted (transition).

    Note: this gait is a rough starting point — the robot currently
    moves backward with these values.
    """
    return [
        # Phase 1
        np.array([
            0.0,  0.728282, -1.750720,  # FR: planted
            0.0,  1.458392, -2.395579,  # FL: swinging
            0.0,  1.458392, -2.395579,  # RR: swinging
            0.0,  0.728282, -1.750720,  # RL: planted
        ]),
        # Phase 2
        np.array([
            0.0,  0.645511, -1.728361,  # FR
            0.0,  1.022439, -1.750720,  # FL
            0.0,  1.022439, -1.750720,  # RR
            0.0,  0.645511, -1.728361,  # RL
        ]),
        # Phase 3
        np.array([
            0.0,  1.458392, -2.395579,  # FR: swinging
            0.0,  0.956029, -1.764179,  # FL: planted
            0.0,  0.956029, -1.764179,  # RR: planted
            0.0,  1.458392, -2.395579,  # RL: swinging
        ]),
        # Phase 4
        np.array([
            0.0,  1.022439, -1.750720,  # FR
            0.0,  0.884337, -1.768673,  # FL
            0.0,  0.884337, -1.768673,  # RR
            0.0,  1.022439, -1.750720,  # RL
        ]),
    ]