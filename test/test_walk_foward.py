"""Test manually designed conservative walk gaits.

Tries small, symmetric trot patterns designed for stability over speed.

Usage:
    cd primitive_go2
    python test/test_walk_manual.py
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path

from env.env import Go2Env
from world.robot_api import RobotAPI
from world.kinematics import Go2Kinematics
from runner.recorder import RenderingEnv


# Neutral positions
# Front: foot_x = +0.1934, foot_z = -0.27
# Rear:  foot_x = -0.1934, foot_z = -0.27

# For forward walking:
#   - Planted leg: foot_x shifts FORWARD (more positive for front, less negative for rear)
#     This pushes the body forward relative to the planted foot.
#   - Swing leg: foot_x shifts BACKWARD (less positive for front, more negative for rear)
#     and foot_z lifts up (less negative) for ground clearance.
#
# After the step, roles swap.

def make_trot_phases(step_size, lift_height):
    """Create a 4-phase trot with given step size and lift height.
    
    step_size:   how far each foot moves from neutral (meters)
    lift_height: how much the swing foot lifts from -0.27 (meters, positive = up)
    """
    # Neutral
    fn = 0.1934   # front neutral foot_x
    rn = -0.1934  # rear neutral foot_x
    zn = -0.27    # neutral foot_z (planted)
    zl = zn + lift_height  # swing foot_z (lifted)

    # Phase 1: FR+RL planted (pushed forward), FL+RR swing (reaching back + lifted)
    # "pushed forward" = foot behind body = body moves forward relative to foot
    p1 = np.array([
        [fn + step_size, zn],    # FR: planted, foot forward of neutral
        [fn - step_size, zl],    # FL: swing, foot behind neutral + lifted
        [rn - step_size, zl],    # RR: swing, foot behind neutral + lifted
        [rn + step_size, zn],    # RL: planted, foot forward of neutral
    ])

    # Phase 2: all legs planted, transition
    p2 = np.array([
        [fn + step_size, zn],    # FR: still forward
        [fn - step_size, zn],    # FL: now planted at back position
        [rn - step_size, zn],    # RR: now planted at back position
        [rn + step_size, zn],    # RL: still forward
    ])

    # Phase 3: FL+RR planted (pushed forward), FR+RL swing (reaching back + lifted)
    p3 = np.array([
        [fn - step_size, zl],    # FR: swing, foot behind neutral + lifted
        [fn + step_size, zn],    # FL: planted, foot forward of neutral
        [rn + step_size, zn],    # RR: planted, foot forward of neutral
        [rn - step_size, zl],    # RL: swing, foot behind neutral + lifted
    ])

    # Phase 4: all legs planted, transition
    p4 = np.array([
        [fn - step_size, zn],    # FR: now planted at back position
        [fn + step_size, zn],    # FL: still forward
        [rn + step_size, zn],    # RR: still forward
        [rn - step_size, zn],    # RL: now planted at back position
    ])

    return [p1, p2, p3, p4]


GAITS = {
    # Very conservative: tiny steps
    "tiny_step": {
        "step_size": 0.02,
        "lift_height": 0.08,
        "phase_duration": 0.3,
        "cycles": 8,
    },
    # Small steps
    "small_step": {
        "step_size": 0.03,
        "lift_height": 0.10,
        "phase_duration": 0.3,
        "cycles": 6,
    },
    # Medium steps
    "medium_step": {
        "step_size": 0.04,
        "lift_height": 0.12,
        "phase_duration": 0.25,
        "cycles": 6,
    },
    # Slow and steady
    "slow_steady": {
        "step_size": 0.03,
        "lift_height": 0.10,
        "phase_duration": 0.5,
        "cycles": 5,
    },
}


def run_walk(name, params, kin):
    print(f"\n{'='*60}")
    print(f"  Testing: {name}")
    print(f"  step_size={params['step_size']}, lift={params['lift_height']}, "
          f"phase_dur={params['phase_duration']}, cycles={params['cycles']}")
    print(f"{'='*60}")

    foot_phases = make_trot_phases(params['step_size'], params['lift_height'])

    for i, feet in enumerate(foot_phases):
        print(f"  Phase {i+1}:")
        for leg, lname in enumerate(["FR", "FL", "RR", "RL"]):
            print(f"    {lname}: foot_x={feet[leg,0]:+.4f}, foot_z={feet[leg,1]:+.4f}")

    joint_phases = [kin.policy_to_joints(feet) for feet in foot_phases]

    base_env = Go2Env(
        xml_path="go2/scene.xml",
        control_substeps=4,
        kp=80.0, kd=4.0,
        initial_base_height=0.27,
        initial_angles=[0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
        settle_steps=500,
    )
    base_env.reset()
    env = RenderingEnv(base_env)
    robot = RobotAPI(env)

    for cycle in range(params['cycles']):
        for phase in joint_phases:
            robot.set_joints(phase)
            robot.step(params['phase_duration'])

    state = robot.get_state()

    video_dir = Path("logs/test_walk_manual")
    video_dir.mkdir(parents=True, exist_ok=True)
    env.save_video(video_dir / f"{name}.mp4")
    env.close()

    print(f"\n  Final x={state['x']:+.4f}  h={state['h']:.4f}  "
          f"roll={state['roll']:.4f}  pitch={state['pitch']:.4f}")
    stable = abs(state['roll']) < 0.15 and abs(state['pitch']) < 0.15 and state['h'] > 0.20
    print(f"  Stable: {'YES' if stable else 'NO'}")
    return state


def main():
    print("=" * 60)
    print("  Manual Walk Gait Test")
    print("=" * 60)

    # Get kinematics
    base_env = Go2Env(
        xml_path="go2/scene.xml",
        control_substeps=4,
        kp=80.0, kd=4.0,
        initial_base_height=0.27,
        initial_angles=[0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
        settle_steps=500,
    )
    base_env.reset()
    kin = Go2Kinematics(base_env.model)
    base_env.close()

    results = {}
    for name, params in GAITS.items():
        state = run_walk(name, params, kin)
        results[name] = state

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for name, state in results.items():
        direction = "FORWARD" if state['x'] > 0.01 else "BACKWARD" if state['x'] < -0.01 else "NONE"
        stable = abs(state['roll']) < 0.15 and abs(state['pitch']) < 0.15 and state['h'] > 0.20
        print(f"  {name:20s}  x={state['x']:+.4f}  h={state['h']:.4f}  "
              f"roll={state['roll']:+.4f}  pitch={state['pitch']:+.4f}  "
              f"{'STABLE' if stable else 'UNSTABLE'}  {direction}")
    print(f"\n  Videos saved to: logs/test_walk_manual/")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())