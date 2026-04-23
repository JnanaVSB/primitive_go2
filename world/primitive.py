"""Policy executor.

Takes a Policy from the LLM, orchestrates IK + trajectory + env stepping,
and returns the settled base pose for reward computation.

The env is assumed to be freshly reset and configured with the correct
PD gains for this policy's stiffness mode (handled by the runner).
"""

import numpy as np
import mujoco

from agent.policy import Policy
from world.kinematics import Go2Kinematics
from world.trajectory import make_trajectory, trajectory_duration_to_nsteps


def execute_policy(
    env,
    kin: Go2Kinematics,
    policy: Policy,
    settle_steps_after: int,
) -> dict:
    """Execute one policy and return the settled base pose.

    Assumes env is already reset to the home pose and configured with the
    correct PD gains. The primitive does not mutate gains or reset the env.

    Args:
        env:                the Gym env (already reset, gains already set)
        kin:                Go2Kinematics instance
        policy:             the Policy to execute
        settle_steps_after: number of extra control steps to hold the final
                            target before measuring base state

    Returns:
        dict with 'h', 'roll', 'pitch', and 'foot_world_z' (list of 4 floats).
    """
    # 1. Current joint angles (starting point of trajectory)
    start_joints = env.data.qpos[env._qpos_idx].copy()

    # 2. IK: policy foot targets → target joint angles
    target_joints = kin.policy_to_joints(policy.foot_targets)

    # 3. Build the quintic trajectory
    traj = make_trajectory(start_joints, target_joints, policy.duration)

    # 4. Time per control step (control_substeps physics steps per step call)
    dt_per_control_step = env.model.opt.timestep * env.control_substeps
    n_traj_steps = trajectory_duration_to_nsteps(policy.duration, dt_per_control_step)

    # 5. Execute the trajectory
    for step in range(n_traj_steps):
        t = step * dt_per_control_step
        action = traj(t)
        env.step(action)

    # 6. Settle: hold the final target
    final_target = traj(policy.duration)
    for _ in range(settle_steps_after):
        env.step(final_target)

    # 7. Read base state and foot positions
    return _extract_base_pose(env, kin)


def _extract_base_pose(env, kin: Go2Kinematics) -> dict:
    """Read base height, roll, pitch, and foot world z from env's MuJoCo data.

    Height is world-frame z of the base. Roll and pitch are extracted from
    the base quaternion using standard ZYX Euler conventions.
    Foot world z is read directly from MuJoCo's geom positions in world frame,
    so it is correct regardless of body tilt.
    """
    h = float(env.data.qpos[2])
    quat_wxyz = env.data.qpos[3:7]  # MuJoCo convention: w, x, y, z

    roll, pitch = _quat_to_roll_pitch(quat_wxyz)

    # Read foot z directly from MuJoCo in world frame
    foot_names = ['FR', 'FL', 'RR', 'RL']
    foot_world_z = []
    for name in foot_names:
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        z = float(env.data.geom_xpos[geom_id][2])  # world z
        foot_world_z.append(z)

    return {
        'h': h,
        'roll': float(roll),
        'pitch': float(pitch),
        'foot_world_z': foot_world_z,
    }


def _quat_to_roll_pitch(quat_wxyz: np.ndarray) -> tuple[float, float]:
    """Convert MuJoCo quaternion (w, x, y, z) to roll, pitch in radians.

    Standard ZYX Euler decomposition. Yaw is discarded (not needed for reward).
    """
    w, x, y, z = quat_wxyz

    # Roll (rotation about x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation about y-axis)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)  # gimbal lock guard
    else:
        pitch = np.arcsin(sinp)

    return roll, pitch