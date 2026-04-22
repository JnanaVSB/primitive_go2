"""Quintic spline trajectory generation for joint-space motion.

Given start joint angles, target joint angles, and a duration, produces a
smooth trajectory from start to target with zero velocity and zero acceleration
at both endpoints. This avoids torque spikes at motion start/end when fed to a
PD controller.

The quintic profile s(τ) = 10τ³ - 15τ⁴ + 6τ⁵ maps τ ∈ [0, 1] to a smooth
s ∈ [0, 1] with s(0)=s'(0)=s''(0)=0 and s(1)=1, s'(1)=s''(1)=0.

Pure math. No MuJoCo, no env dependency.
"""

from typing import Callable
import numpy as np


def quintic_profile(tau: float) -> float:
    """Normalized quintic smoothstep: tau ∈ [0, 1] -> s ∈ [0, 1].

    Zero velocity and acceleration at both endpoints.
    """
    tau = float(np.clip(tau, 0.0, 1.0))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def make_trajectory(
    start: np.ndarray,
    target: np.ndarray,
    duration: float,
) -> Callable[[float], np.ndarray]:
    """Build a trajectory function from start to target over `duration` seconds.

    Args:
        start:    joint angle vector at t=0 (shape (N,))
        target:   joint angle vector at t=duration (shape (N,))
        duration: total motion time in seconds (must be > 0)

    Returns:
        A function trajectory(t) -> joint_angles that returns the interpolated
        joint angles at time t. For t <= 0 returns start. For t >= duration
        returns target.

    Example:
        traj = make_trajectory(home_angles, target_angles, duration=2.0)
        for step in range(n_steps):
            t = step * dt
            env.step(traj(t))
    """
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert start.shape == target.shape, (
        f"start and target must have the same shape, got {start.shape} vs {target.shape}"
    )
    assert duration > 0, f"duration must be positive, got {duration}"

    delta = target - start

    def trajectory(t: float) -> np.ndarray:
        tau = t / duration
        s = quintic_profile(tau)
        return start + s * delta

    return trajectory


def trajectory_duration_to_nsteps(duration: float, dt: float) -> int:
    """Convert duration (seconds) to number of control steps.

    Args:
        duration: motion duration in seconds
        dt:       time per control step in seconds
                  (e.g., model.opt.timestep * control_substeps)

    Returns:
        Number of steps to march the trajectory. Rounds up.
    """
    return int(np.ceil(duration / dt))