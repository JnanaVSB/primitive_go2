"""Hand-authored policy validation (no LLM).

Executes manually-designed policies for sit, stand, and lay against the env
with the MuJoCo viewer attached. Prints the resulting base state and reward
vs. the target from the handoff doc.

Use this to verify the primitive + IK + trajectory produce the intended target
poses before introducing the LLM. Iterate on the foot positions below until
the poses look right.

Run:
    python -m test.test_nollm              # run all three
    python -m test.test_nollm stand        # run just one
    python -m test.test_nollm sit lay      # run multiple
"""

import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from env.env import Go2Env
from env.reward import compute_pose_reward
from world.kinematics import Go2Kinematics
from world.trajectory import make_trajectory, trajectory_duration_to_nsteps
from agent.policy import Policy


XML_PATH = "go2/scene.xml"

# Home keyframe from go2.xml
HOME_BASE_HEIGHT = 0.27
HOME_ANGLES = [
    0.0, 0.9, -1.8,   # FR
    0.0, 0.9, -1.8,   # FL
    0.0, 0.9, -1.8,   # RR
    0.0, 0.9, -1.8,   # RL
]

# Target body poses from the handoff doc (h, roll, pitch)
TARGETS = {
    'stand': {'h': 0.27, 'roll': 0.0, 'pitch': 0.0},
    'sit':   {'h': 0.17, 'roll': 0.0, 'pitch': -0.3},
    'lay':   {'h': 0.08, 'roll': 0.0, 'pitch': 0.0},
}

# Hand-authored foot targets in base_link frame.
# Row order matches LEG_NAMES = ['FR', 'FL', 'RR', 'RL'].
# Each row is (foot_x, foot_z) in meters.
#
# STAND: home pose — feet directly under hips, legs extended.
# SIT:   rear feet pulled forward and up → body tilts back, lowers.
# LAY:   all feet pulled in toward body → belly drops to the floor.
#
# Tweak these by hand until the viewer shows the right pose and reward
# approaches zero.
HAND_POLICIES = {
    'stand': Policy(
        foot_targets=np.array([
            [ 0.1934, -0.27],   # FR
            [ 0.1934, -0.27],   # FL
            [-0.1934, -0.27],   # RR
            [-0.1934, -0.27],   # RL
        ]),
        duration=2.0,
        stiffness='normal',
    ),
    'sit': Policy(
        foot_targets=np.array([
            [ 0.1934, -0.40],   # FR — front legs extended (unchanged)
            [ 0.1934, -0.40],   # FL
            [-0.12,   -0.05],   # RR — rear feet tucked close + high
            [-0.12,   -0.05],   # RL
        ]),
        duration=2.5,
        stiffness='normal',
    ),
    'lay': Policy(
        foot_targets=np.array([
            [ 0.15, -0.1],    # FR — feet close to hips but still extended down
            [ 0.15, -0.1],    # FL
            [-0.15, -0.1],    # RR
            [-0.15, -0.1],    # RL
        ]),
        duration=2.5,
        stiffness='normal',
    ),
}

# Stiffness mode → (kp, kd)
STIFFNESS_MODES = {
    'soft':   (40.0, 3.0),
    'normal': (80.0, 4.0),
    'stiff':  (150.0, 8.0),
}

SETTLE_STEPS_AFTER = 500


def extract_base_pose(env):
    """Read h, roll, pitch from env's MuJoCo data."""
    from world.primitive import _quat_to_roll_pitch
    h = float(env.data.qpos[2])
    quat = env.data.qpos[3:7]
    roll, pitch = _quat_to_roll_pitch(quat)
    return {'h': h, 'roll': float(roll), 'pitch': float(pitch)}


def run_policy_with_viewer(task_name: str, policy: Policy, target: dict):
    """Execute a hand-authored policy with the MuJoCo viewer attached.

    Slows the sim to real-time so motion is watchable.
    """
    kp, kd = STIFFNESS_MODES[policy.stiffness]
    env = Go2Env(
        xml_path=XML_PATH,
        control_substeps=4,
        kp=kp,
        kd=kd,
        initial_base_height=HOME_BASE_HEIGHT,
        initial_angles=HOME_ANGLES,
        settle_steps=500,
    )
    env.reset()
    kin = Go2Kinematics(env.model)

    # Build trajectory
    start_joints = env.data.qpos[env._qpos_idx].copy()
    target_joints = kin.policy_to_joints(policy.foot_targets)
    traj = make_trajectory(start_joints, target_joints, policy.duration)
    dt = env.model.opt.timestep * env.control_substeps
    n_steps = trajectory_duration_to_nsteps(policy.duration, dt)

    print(f"\n{'=' * 60}")
    print(f"Task: {task_name.upper()}")
    print(f"Target: h={target['h']:.3f}, roll={target['roll']:.3f}, "
          f"pitch={target['pitch']:.3f}")
    print(f"Stiffness: {policy.stiffness} (Kp={kp}, Kd={kd})")
    print(f"Duration: {policy.duration}s, trajectory steps: {n_steps}")
    print(f"{'=' * 60}")
    print("Opening viewer. Close the window or press ESC to continue to next task.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Hold home pose briefly so we see the start
        for _ in range(100):
            if not viewer.is_running():
                break
            env.step(start_joints)
            viewer.sync()
            time.sleep(dt)

        # Execute the trajectory
        for step in range(n_steps):
            if not viewer.is_running():
                break
            t = step * dt
            env.step(traj(t))
            viewer.sync()
            time.sleep(dt)

        # Settle at the final target
        final_target = traj(policy.duration)
        for _ in range(SETTLE_STEPS_AFTER):
            if not viewer.is_running():
                break
            env.step(final_target)
            viewer.sync()
            time.sleep(dt)

        # Let the user inspect the final pose
        base_state = extract_base_pose(env)
        reward = compute_pose_reward(base_state, target)

        print(f"\nResult:")
        print(f"  h:     {base_state['h']:.4f}  (target {target['h']:.4f}, "
              f"Δ={base_state['h'] - target['h']:+.4f})")
        print(f"  roll:  {base_state['roll']:+.4f}  (target {target['roll']:+.4f}, "
              f"Δ={base_state['roll'] - target['roll']:+.4f})")
        print(f"  pitch: {base_state['pitch']:+.4f}  (target {target['pitch']:+.4f}, "
              f"Δ={base_state['pitch'] - target['pitch']:+.4f})")
        print(f"  reward: {reward:.4f}")
        print("\nHolding final pose. Close viewer or press ESC to continue.")

        # Keep viewer alive until user closes it
        while viewer.is_running():
            env.step(final_target)
            viewer.sync()
            time.sleep(dt)


def main():
    tasks = sys.argv[1:] if len(sys.argv) > 1 else list(HAND_POLICIES.keys())

    for task in tasks:
        if task not in HAND_POLICIES:
            print(f"Unknown task: {task}. Valid: {list(HAND_POLICIES.keys())}")
            continue
        run_policy_with_viewer(task, HAND_POLICIES[task], TARGETS[task])

    print("\nDone.")


if __name__ == '__main__':
    main()