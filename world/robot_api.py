"""Robot API for LLM-generated code.

Clean interface that wraps Go2Env for the code-as-policy system.
The LLM's code interacts with the robot exclusively through this API.

The LLM controls:
    - Which joint angles to target (via set_joints)
    - How long to interpolate to those angles (via step)
    - When to read state for reactive decisions (via get_state)
    - When to mark task completion (via checkpoint)

The infrastructure handles:
    - Quintic trajectory interpolation (smooth, no torque spikes)
    - PD control and physics stepping
    - Frame capture for video recording
    - Per-task reward computation at checkpoints

Usage (from LLM-generated code):
    robot.set_joints(get_stand_pose())
    robot.step(2.0)
    robot.checkpoint("stand")

    walk = get_walk_planner(stride_length=0.10)
    planner = walk['planner']
    period = walk['cycle_period']
    for i in range(100):
        phi = (i * robot.dt / period) % 1.0
        robot.set_joints(walk_step(planner, phi))
        robot.step(robot.dt)
    robot.checkpoint("walk_5m")

    robot.set_joints(get_sit_pose())
    robot.step(2.0)
    robot.checkpoint("sit")
"""

import numpy as np

from world.trajectory import make_trajectory, trajectory_duration_to_nsteps


class RobotAPI:
    """LLM-facing robot interface.

    Wraps a Go2Env (or RenderingEnv) and provides set_joints / step /
    get_state / checkpoint. All trajectory interpolation happens inside step().

    Args:
        env: a Go2Env or RenderingEnv instance (already reset).
    """

    def __init__(self, env):
        self._env = env
        self._target_joints = env.data.qpos[env._qpos_idx].copy()
        self._checkpoints = []

    @property
    def dt(self) -> float:
        """Simulation time step per control step in seconds.

        Use this for walk gait loops:
            phi = (step * robot.dt / cycle_period) % 1.0
        """
        return self._env.model.opt.timestep * self._env.control_substeps

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

    def step_direct(self, joint_angles: np.ndarray = None):
        """Send joint angles directly to the PD controller for one control step.

        Unlike step(), this does NOT interpolate. It sends the joint angles
        straight to the env and advances by one dt. Use this for walk gait
        loops where the planner provides smooth targets every timestep.

        Args:
            joint_angles: 12-dim array of joint angle targets. If None,
                          uses the last targets set via set_joints().
        """
        if joint_angles is not None:
            self._target_joints = np.asarray(joint_angles, dtype=np.float64).flatten()[:12]
        self._env.step(self._target_joints)

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

    def checkpoint(self, name: str):
        """Record the current state as a named checkpoint.

        Call this after completing each subtask. The runner uses
        checkpoints to compute per-task rewards.

        Args:
            name: label for this checkpoint (e.g. "walk_5m", "sit", "stand").
        """
        self._checkpoints.append({
            'name': name,
            'state': self.get_state(),
        })

    @property
    def checkpoints(self) -> list[dict]:
        """All recorded checkpoints as a list of {'name': str, 'state': dict}."""
        return self._checkpoints