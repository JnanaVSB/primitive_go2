"""Policy dataclass — the LLM's output schema.

A single Policy represents one motion target: where the feet should go,
how fast to get there, and how stiffly to track the trajectory.

This is the type passed from the LLM parser to the primitive executor.
It's also what gets serialized to the trial history log.
"""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np


Stiffness = Literal['soft', 'normal', 'stiff']


@dataclass
class Policy:
    """One motion target for the Go2.

    Attributes:
        foot_targets: shape (4, 2). Row order matches LEG_NAMES (FR, FL, RR, RL).
                      Each row is (foot_x, foot_z) in base_link frame, meters.
        duration:     motion duration in seconds. Fixed at 5.0 by default.
                      The LLM does not control this parameter.
        stiffness:    controller stiffness mode: 'soft' | 'normal' | 'stiff'.
                      Resolves to (kp, kd) via config lookup at env construction.
    """
    foot_targets: np.ndarray
    duration: float = 5.0
    stiffness: Stiffness = 'normal'

    def __post_init__(self):
        self.foot_targets = np.asarray(self.foot_targets, dtype=np.float64)
        if self.foot_targets.shape != (4, 2):
            raise ValueError(
                f"foot_targets must have shape (4, 2), got {self.foot_targets.shape}"
            )
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.stiffness not in ('soft', 'normal', 'stiff'):
            raise ValueError(
                f"stiffness must be 'soft'|'normal'|'stiff', got {self.stiffness!r}"
            )

    def to_dict(self) -> dict:
        """Serialize for trial history logging."""
        return {
            'foot_targets': self.foot_targets.tolist(),
            'duration': self.duration,
            'stiffness': self.stiffness,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Policy':
        """Deserialize from trial history."""
        return cls(
            foot_targets=np.asarray(d['foot_targets']),
            duration=d['duration'],
            stiffness=d['stiffness'],
        )