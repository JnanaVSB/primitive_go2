"""Pose-distance reward for sit/stand/lay/walk tasks.

R = -(|Δh| + |Δroll| + |Δpitch|) + distance_weight * x_distance

The distance term is zero for stationary tasks (sit, lay, stand) and
positive for movement tasks (walk). The weight is configured per task step.

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
        base_state: {'h': float, 'roll': float, 'pitch': float, 'x': float}
                    — current base pose and x position.
        target:     {'h': float, 'roll': float, 'pitch': float}
                    — desired base pose.
        distance_weight: multiplier for the x distance term.
                    0.0 for stationary tasks, > 0 for walk.

    Returns:
        Scalar reward. Pose terms are always <= 0.
        Distance term can be positive (moved forward) or negative (moved backward).
    """
    dh = abs(base_state['h'] - target['h'])
    droll = abs(base_state['roll'] - target['roll'])
    dpitch = abs(base_state['pitch'] - target['pitch'])

    pose_reward = -(dh + droll + dpitch)

    distance_reward = distance_weight * base_state.get('x', 0.0)

    return pose_reward + distance_reward