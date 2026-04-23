"""Pose-distance reward for sit/stand/lay tasks.

R = -(|Δh| + |Δroll| + |Δpitch|)

With termination: if any foot is above max_foot_z from the ground,
the trial is a failure and returns a fixed penalty.

Pure functions. No env or MuJoCo dependency.
All angles in radians, height in meters.
"""

# Penalty returned when a foot exceeds the height threshold.
TERMINATION_PENALTY = -10.0


def check_foot_termination(foot_world_z: list[float], max_foot_z: float) -> bool:
    """Check if any foot is too high off the ground.

    Args:
        foot_world_z: list of 4 foot heights in world frame
                      (body height + foot z in body frame).
        max_foot_z:   maximum allowed foot height from ground.
                      If any foot exceeds this, the trial fails.

    Returns:
        True if any foot is above the threshold (trial should terminate).
        False if all feet are within range (trial is OK).
    """
    for z in foot_world_z:
        if z > max_foot_z:
            return True
    return False


def compute_pose_reward(base_state: dict, target: dict) -> float:
    """Scalar reward from base pose deviation.

    Args:
        base_state: {'h': float, 'roll': float, 'pitch': float} — current base pose.
        target:     {'h': float, 'roll': float, 'pitch': float} — desired base pose.

    Returns:
        Scalar reward <= 0. Perfect match = 0.
    """
    dh = abs(base_state['h'] - target['h'])
    droll = abs(base_state['roll'] - target['roll'])
    dpitch = abs(base_state['pitch'] - target['pitch'])
    return -(dh + droll + dpitch)