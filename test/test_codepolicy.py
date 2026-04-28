"""End-to-end dummy test for the code-as-policy pipeline.

Bypasses the LLM entirely — uses hardcoded policy code to verify the
full pipeline: env setup → RobotAPI → code executor → reward computation.

Usage:
    cd primitive_go2
    python test/test_codepolicy.py
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from dataclasses import asdict

from config import load_config
from env.env import Go2Env
from env.reward import compute_pose_reward
from world.robot_api import RobotAPI
from runner.code_executor import execute_policy_code
from runner.recorder import RenderingEnv


# Hardcoded policy code snippets to test each primitive
TEST_CASES = {
    "stand": """
robot.set_joints(get_stand_pose())
robot.step(2.0)
""",
    "sit": """
robot.set_joints(get_sit_pose())
robot.step(2.0)
""",
    "lay": """
robot.set_joints(get_lay_pose())
robot.step(2.0)
""",
    "walk": """
phases = get_walk_phases()
for cycle in range(3):
    for phase in phases:
        robot.set_joints(phase)
        robot.step(0.3)
""",
    "sit_then_stand": """
robot.set_joints(get_sit_pose())
robot.step(2.0)
robot.set_joints(get_stand_pose())
robot.step(2.0)
""",
}

# Targets for reward computation
TARGETS = {
    "stand": {"h": 0.27, "roll": 0.0, "pitch": 0.0},
    "sit":   {"h": 0.17, "roll": 0.0, "pitch": -0.3},
    "lay":   {"h": 0.05, "roll": 0.0, "pitch": 0.0},
    "walk":  {"h": 0.27, "roll": 0.0, "pitch": 0.0},
    "sit_then_stand": {"h": 0.27, "roll": 0.0, "pitch": 0.0},
}


def run_test(name, code, target):
    """Run one test case through the full pipeline."""
    print(f"\n{'='*60}")
    print(f"  Testing: {name}")
    print(f"{'='*60}")
    print(f"  Code:\n{code.strip()}")
    print(f"  Target: {target}")

    # Create env
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

    # Execute code
    result = execute_policy_code(code, robot)

    # Read state and compute reward
    state = robot.get_state()
    reward = compute_pose_reward(state, target)

    # Save video
    video_dir = Path("logs/test_codepolicy")
    video_dir.mkdir(parents=True, exist_ok=True)
    env.save_video(video_dir / f"{name}.mp4")
    env.close()

    # Report
    print(f"\n  Execution: {'SUCCESS' if result.success else 'FAILED'}")
    if not result.success:
        print(f"  Error: {result.error}")
    print(f"  State:  h={state['h']:.4f} roll={state['roll']:.4f} pitch={state['pitch']:.4f} x={state['x']:.4f}")
    print(f"  Reward: {reward:.4f}")

    return result.success, reward


def main():
    print("=" * 60)
    print("  Code-as-Policy Pipeline Test (no LLM)")
    print("=" * 60)

    results = {}
    for name in TEST_CASES:
        success, reward = run_test(name, TEST_CASES[name], TARGETS[name])
        results[name] = (success, reward)

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    all_passed = True
    for name, (success, reward) in results.items():
        status = "OK" if success else "EXEC_FAIL"
        print(f"  {name:20s}  {status:10s}  reward={reward:.4f}")
        if not success:
            all_passed = False

    print(f"\n  All executed: {'YES' if all_passed else 'NO'}")
    print(f"  Videos saved to: logs/test_codepolicy/")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())