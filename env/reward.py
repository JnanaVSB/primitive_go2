"""Pose-distance reward for sit/stand/lay tasks.

R = -(|Δh| + |Δroll| + |Δpitch|)

Pure function. No env or MuJoCo dependency.
All angles in radians, height in meters.
"""


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