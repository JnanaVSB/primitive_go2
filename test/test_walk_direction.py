"""Test different walk direction fixes.

Runs three walk variants and saves videos so you can compare:
  1. Original phases (moves backward)
  2. Reversed phase order (4-3-2-1)
  3. Swapped front/rear leg assignments in each phase

Usage:
    cd primitive_go2
    python test/test_walk_direction.py
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path

from env.env import Go2Env
from world.robot_api import RobotAPI
from runner.code_executor import execute_policy_code
from runner.recorder import RenderingEnv


# Original phases (from primitives.py — moves backward)
ORIGINAL_PHASES = [
    # Phase 1
    np.array([
        0.0,  0.728282, -1.750720,  # FR
        0.0,  1.458392, -2.395579,  # FL
        0.0,  1.458392, -2.395579,  # RR
        0.0,  0.728282, -1.750720,  # RL
    ]),
    # Phase 2
    np.array([
        0.0,  0.645511, -1.728361,  # FR
        0.0,  1.022439, -1.750720,  # FL
        0.0,  1.022439, -1.750720,  # RR
        0.0,  0.645511, -1.728361,  # RL
    ]),
    # Phase 3
    np.array([
        0.0,  1.458392, -2.395579,  # FR
        0.0,  0.956029, -1.764179,  # FL
        0.0,  0.956029, -1.764179,  # RR
        0.0,  1.458392, -2.395579,  # RL
    ]),
    # Phase 4
    np.array([
        0.0,  1.022439, -1.750720,  # FR
        0.0,  0.884337, -1.768673,  # FL
        0.0,  0.884337, -1.768673,  # RR
        0.0,  1.022439, -1.750720,  # RL
    ]),
]

# Variant 1: Reversed phase order
REVERSED_PHASES = list(reversed(ORIGINAL_PHASES))

# Variant 2: Swap front and rear legs within each phase
# FR(0:3) <-> RR(6:9), FL(3:6) <-> RL(9:12)
SWAPPED_PHASES = []
for phase in ORIGINAL_PHASES:
    swapped = np.zeros(12)
    swapped[0:3] = phase[6:9]    # FR gets RR's values
    swapped[3:6] = phase[9:12]   # FL gets RL's values
    swapped[6:9] = phase[0:3]    # RR gets FR's values
    swapped[9:12] = phase[3:6]   # RL gets FL's values
    SWAPPED_PHASES.append(swapped)


WALK_CODE_TEMPLATE = """
phases = PHASES
for cycle in range(5):
    for phase in phases:
        robot.set_joints(phase)
        robot.step(0.3)
"""


def run_walk(name, phases):
    print(f"\n{'='*60}")
    print(f"  Testing: {name}")
    print(f"{'='*60}")

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

    # Run walk
    for cycle in range(5):
        for phase in phases:
            robot.set_joints(phase)
            robot.step(0.3)

    state = robot.get_state()

    video_dir = Path("logs/test_walk_direction")
    video_dir.mkdir(parents=True, exist_ok=True)
    env.save_video(video_dir / f"{name}.mp4")
    env.close()

    print(f"  Final x={state['x']:+.4f} (positive=forward, negative=backward)")
    print(f"  Final h={state['h']:.4f}  roll={state['roll']:.4f}  pitch={state['pitch']:.4f}")
    return state['x']


def main():
    print("=" * 60)
    print("  Walk Direction Test")
    print("=" * 60)

    results = {}
    results['original'] = run_walk('original', ORIGINAL_PHASES)
    results['reversed'] = run_walk('reversed', REVERSED_PHASES)
    results['swapped'] = run_walk('swapped_front_rear', SWAPPED_PHASES)

    print(f"\n{'='*60}")
    print("  Summary (positive x = forward)")
    print(f"{'='*60}")
    for name, x in results.items():
        direction = "FORWARD" if x > 0.01 else "BACKWARD" if x < -0.01 else "NO MOVEMENT"
        print(f"  {name:25s}  x={x:+.4f}  {direction}")
    print(f"\n  Videos saved to: logs/test_walk_direction/")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())
