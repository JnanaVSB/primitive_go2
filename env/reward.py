"""Pose-distance reward for sit/stand/lay/walk/turn tasks.

R = -(|Δh| + |Δroll| + |Δpitch| + |Δyaw|) + distance_weight * x_distance

The yaw term is included when the target specifies a yaw value.
For tasks without a yaw target, it is ignored (backward compatible).
The distance term is zero for stationary tasks and positive for movement tasks.

Pure function. No env or MuJoCo dependency.
All angles in radians, height in meters, distance in meters.
"""


def compute_pose_reward(
    base_state: dict,
    target: dict,
    distance_weight: float = 0.0,
) -> float:
    """Scalar reward from base pose deviation and distance traveled.

    Args:
        base_state: {'h': float, 'roll': float, 'pitch': float, 'yaw': float, 'x': float}
                    — current base pose, yaw, and x position.
        target:     {'h': float, 'roll': float, 'pitch': float}
                    — desired base pose. Optionally includes 'yaw'.
        distance_weight: multiplier for the x distance term.
                    0.0 for stationary tasks, > 0 for walk.

    Returns:
        Scalar reward. Pose terms are always <= 0.
        Distance term can be positive (moved forward) or negative (moved backward).
    """
    dh = abs(base_state['h'] - target['h'])
    droll = abs(base_state['roll'] - target['roll'])
    dpitch = abs(base_state['pitch'] - target['pitch'])

    dyaw = abs(base_state.get('yaw', 0.0) - target.get('yaw', 0.0))
    # Wrap to [-pi, pi] so turning 359° isn't penalized as much as 1°
    if dyaw > 3.14159265:
        dyaw = 2 * 3.14159265 - dyaw

    pose_reward = -(dh + droll + dpitch + dyaw)

    distance_reward = distance_weight * base_state.get('x', 0.0)

    return pose_reward + distance_reward