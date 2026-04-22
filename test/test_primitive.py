"""Tests for world/primitive.py.

Exercises the full execute_policy pipeline: IK + trajectory + env stepping
+ settle + base state extraction. Uses the real MuJoCo model.
"""

import numpy as np
import pytest

from env.env import Go2Env
from world.kinematics import Go2Kinematics
from world.primitive import execute_policy, _quat_to_roll_pitch
from agent.policy import Policy


XML_PATH = "go2/scene.xml"

# Home keyframe values from go2.xml
HOME_BASE_HEIGHT = 0.27
HOME_THIGH = 0.9
HOME_CALF = -1.8
HOME_ANGLES = [
    0.0, HOME_THIGH, HOME_CALF,   # FR
    0.0, HOME_THIGH, HOME_CALF,   # FL
    0.0, HOME_THIGH, HOME_CALF,   # RR
    0.0, HOME_THIGH, HOME_CALF,   # RL
]


@pytest.fixture
def env():
    """Fresh env at home pose."""
    e = Go2Env(
        xml_path=XML_PATH,
        control_substeps=4,
        kp=80.0,
        kd=4.0,
        initial_base_height=HOME_BASE_HEIGHT,
        initial_angles=HOME_ANGLES,
        settle_steps=500,
    )
    e.reset()
    return e


@pytest.fixture
def kin(env):
    return Go2Kinematics(env.model)


class TestQuatConversion:
    def test_identity_quat(self):
        """Identity quaternion gives zero roll and pitch."""
        roll, pitch = _quat_to_roll_pitch(np.array([1.0, 0.0, 0.0, 0.0]))
        assert abs(roll) < 1e-10
        assert abs(pitch) < 1e-10

    def test_pure_roll(self):
        """Quaternion for 30° roll gives roll=0.5236 rad, pitch=0."""
        angle = np.pi / 6  # 30°
        # Rotation about x-axis: (cos(a/2), sin(a/2), 0, 0)
        q = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        roll, pitch = _quat_to_roll_pitch(q)
        assert abs(roll - angle) < 1e-6
        assert abs(pitch) < 1e-6

    def test_pure_pitch(self):
        """Quaternion for -20° pitch gives roll=0, pitch=-0.3491 rad."""
        angle = -np.pi / 9  # -20°
        # Rotation about y-axis: (cos(a/2), 0, sin(a/2), 0)
        q = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
        roll, pitch = _quat_to_roll_pitch(q)
        assert abs(roll) < 1e-6
        assert abs(pitch - angle) < 1e-6


class TestExecutePolicy:
    def test_stay_at_home_produces_home_pose(self, env, kin):
        """If the policy targets the current (home) foot positions,
        the robot should barely move and end up near home."""
        current_joints = env.data.qpos[env._qpos_idx].copy()
        current_feet = kin.forward_kinematics(current_joints)

        policy = Policy(
            foot_targets=current_feet,
            duration=1.0,
            stiffness='normal',
        )
        base_state = execute_policy(env, kin, policy, settle_steps_after=500)

        # Base should be near home height with minimal tilt
        assert abs(base_state['h'] - HOME_BASE_HEIGHT) < 0.03, (
            f"height {base_state['h']:.4f} deviates too much from home {HOME_BASE_HEIGHT}"
        )
        assert abs(base_state['roll']) < 0.1, f"roll {base_state['roll']:.4f}"
        assert abs(base_state['pitch']) < 0.1, f"pitch {base_state['pitch']:.4f}"

    def test_return_has_expected_keys(self, env, kin):
        """execute_policy returns a dict with h, roll, pitch."""
        current_joints = env.data.qpos[env._qpos_idx].copy()
        current_feet = kin.forward_kinematics(current_joints)
        policy = Policy(foot_targets=current_feet, duration=0.5)

        result = execute_policy(env, kin, policy, settle_steps_after=100)

        assert set(result.keys()) == {'h', 'roll', 'pitch'}
        assert all(isinstance(v, float) for v in result.values())

    def test_lowering_feet_reduces_height(self, env, kin):
        """Pulling all feet closer to the body (smaller |foot_z|) should
        lower the base — the robot crouches."""
        current_joints = env.data.qpos[env._qpos_idx].copy()
        current_feet = kin.forward_kinematics(current_joints)

        # Raise all feet 5 cm (less negative foot_z = feet closer to body)
        crouched_feet = current_feet.copy()
        crouched_feet[:, 1] += 0.05

        policy = Policy(foot_targets=crouched_feet, duration=1.5, stiffness='normal')
        base_state = execute_policy(env, kin, policy, settle_steps_after=500)

        # Body should be meaningfully lower than home
        assert base_state['h'] < HOME_BASE_HEIGHT - 0.02, (
            f"expected crouch (h < {HOME_BASE_HEIGHT - 0.02:.3f}), "
            f"got h={base_state['h']:.4f}"
        )

    def test_duration_affects_nothing_at_steady_state(self, env, kin):
        """Short vs long duration — after settle, both should reach similar
        final pose for the same target."""
        current_joints = env.data.qpos[env._qpos_idx].copy()
        current_feet = kin.forward_kinematics(current_joints)

        policy_fast = Policy(foot_targets=current_feet, duration=0.5)
        state_fast = execute_policy(env, kin, policy_fast, settle_steps_after=500)

        env.reset()
        policy_slow = Policy(foot_targets=current_feet, duration=2.0)
        state_slow = execute_policy(env, kin, policy_slow, settle_steps_after=500)

        # Both should settle to similar base pose
        assert abs(state_fast['h'] - state_slow['h']) < 0.02
        assert abs(state_fast['pitch'] - state_slow['pitch']) < 0.05

# prints the output at each time step and run it using this command: pytest test/test_primitive.py::test_diagnostic -v -s

def test_diagnostic(env, kin):
    """Diagnostic: print state at each phase."""
    print(f"\n=== After reset ===")
    print(f"qpos[0:7] (base): {env.data.qpos[0:7]}")
    print(f"joint angles (JOINT_NAMES order): {env.data.qpos[env._qpos_idx]}")
    print(f"base height: {env.data.qpos[2]:.4f}")

    current_joints = env.data.qpos[env._qpos_idx].copy()
    current_feet = kin.forward_kinematics(current_joints)
    target_joints = kin.policy_to_joints(current_feet)

    print(f"\n=== IK round-trip check ===")
    print(f"current_joints:  {current_joints}")
    print(f"target_joints:   {target_joints}")
    print(f"max diff:        {np.abs(current_joints - target_joints).max():.6e}")

    # Step once and see what happens
    print(f"\n=== After one env.step with current joints as target ===")
    env.step(current_joints)
    print(f"qpos[0:7]: {env.data.qpos[0:7]}")
    print(f"joint angles: {env.data.qpos[env._qpos_idx]}")
    print(f"base height: {env.data.qpos[2]:.4f}")

    # Step 10 more times
    for i in range(10):
        env.step(current_joints)
    print(f"\n=== After 10 more env.step calls ===")
    print(f"qpos[0:7]: {env.data.qpos[0:7]}")
    print(f"base height: {env.data.qpos[2]:.4f}")