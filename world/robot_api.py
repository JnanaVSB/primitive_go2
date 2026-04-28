"""Robot API for LLM-generated code.

Clean interface that wraps Go2Env for the code-as-policy system.
The LLM's code interacts with the robot exclusively through this API.

The LLM controls:
    - Which joint angles to target (via set_joints)
    - How long to interpolate to those angles (via step)
    - When to read state for reactive decisions (via get_state)

The infrastructure handles:
    - Quintic trajectory interpolation (smooth, no torque spikes)
    - PD control and physics stepping
    - Frame capture for video recording

Usage (from LLM-generated code):
    robot.set_joints(get_stand_pose())
    robot.step(2.0)             # interpolate to stand over 2 seconds

    robot.set_joints(get_sit_pose())
    robot.step(0.3)             # fast transition — snap to sit

    state = robot.get_state()   # check where we are
    if state['h'] > 0.2:
        robot.set_joints(get_lay_pose())
        robot.step(1.0)
"""

import numpy as np

from world.trajectory import make_trajectory, trajectory_duration_to_nsteps


class RobotAPI:
    """LLM-facing robot interface.

    Wraps a Go2Env (or RenderingEnv) and provides set_joints / step /
    get_state. All trajectory interpolation happens inside step().

    Args:
        env: a Go2Env or RenderingEnv instance (already reset).
    """

    def __init__(self, env):
        self._env = env
        self._target_joints = env.data.qpos[env._qpos_idx].copy()

    def set_joints(self, joint_angles: np.ndarray):
        """Set the target joint angles for the next step() call.

        Args:
            joint_angles: 12-dim array of joint angle targets.
                          Order: [FR_hip, FR_thigh, FR_calf, FL_hip, ..., RL_calf].
                          Typically from a primitive like get_stand_pose().

        Does not move the robot — call step() after to execute the motion.
        """
        self._target_joints = np.asarray(joint_angles, dtype=np.float64).flatten()[:12]

    def step(self, duration: float):
        """Interpolate from current joint positions to the target over duration seconds.

        Uses a quintic spline for smooth acceleration (zero velocity and
        acceleration at both endpoints). The robot's PD controller tracks
        the interpolated trajectory at each control step.

        Args:
            duration: time in seconds for the motion. Short duration = fast
                      motion (e.g. 0.2s for a jump), long = smooth (e.g. 3.0s
                      for a gentle transition).
        """
        if duration <= 0:
            return

        start_joints = self._env.data.qpos[self._env._qpos_idx].copy()
        traj = make_trajectory(start_joints, self._target_joints, duration)

        dt_per_step = self._env.model.opt.timestep * self._env.control_substeps
        n_steps = trajectory_duration_to_nsteps(duration, dt_per_step)

        for i in range(n_steps):
            t = i * dt_per_step
            action = traj(t)
            self._env.step(action)

    def get_state(self) -> dict:
        """Read the current robot state.

        Returns:
            dict with:
                'h':     base height in meters
                'roll':  body roll in radians
                'pitch': body pitch in radians
                'x':     forward position in meters
                'y':     lateral position in meters
        """
        qpos = self._env.data.qpos
        x = float(qpos[0])
        y = float(qpos[1])
        h = float(qpos[2])

        # Quaternion to roll/pitch (MuJoCo convention: w, x, y, z)
        w, qx, qy, qz = qpos[3:7]

        sinr_cosp = 2.0 * (w * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = float(np.arctan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (w * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = float(np.copysign(np.pi / 2.0, sinp))
        else:
            pitch = float(np.arcsin(sinp))

        return {
            'h': h,
            'roll': roll,
            'pitch': pitch,
            'x': x,
            'y': y,
        }