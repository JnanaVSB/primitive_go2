"""Sequential walk: one leg at a time with weight shift.

Weight shift: bend the THREE supporting legs slightly (lower foot_z)
on the opposite side to lean the body away from the leg being lifted.

Usage:
    cd primitive_go2
    python test/test_walk_sequential.py
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


def make_sequential_gait(d, lift_z, crouch_z):
    """Create a sequential gait moving one leg at a time.

    Order: FR, RL, FL, RR (diagonal pattern for stability).
    Before lifting a leg, bend the opposite-side legs to shift weight away.

    d:        step offset to add to foot_x
    lift_z:   foot_z when swinging (e.g. -0.15)
    crouch_z: foot_z for opposite legs during weight shift (e.g. -0.30)
              deeper than ground = legs bend more = body leans away
    """
    fn = 0.1934
    rn = -0.1934
    gz = -0.27

    # Track where each leg's foot_x currently is
    # Starts at neutral, accumulates d each time a leg moves
    fr_x, fl_x, rr_x, rl_x = fn, fn, rn, rn

    phases = []

    # --- Move FR forward ---
    # FR is front-right. Shift weight to LEFT side: bend FL and RL deeper.
    phases.append(np.array([[fr_x, gz], [fl_x, crouch_z], [rr_x, gz], [rl_x, crouch_z]]))
    # Lift FR
    phases.append(np.array([[fr_x, lift_z], [fl_x, crouch_z], [rr_x, gz], [rl_x, crouch_z]]))
    # Move FR forward
    phases.append(np.array([[fr_x + d, lift_z], [fl_x, crouch_z], [rr_x, gz], [rl_x, crouch_z]]))
    # Plant FR, restore others
    fr_x += d
    phases.append(np.array([[fr_x, gz], [fl_x, gz], [rr_x, gz], [rl_x, gz]]))

    # --- Move RL forward ---
    # RL is rear-left. Shift weight to RIGHT side: bend FR and RR deeper.
    phases.append(np.array([[fr_x, crouch_z], [fl_x, gz], [rr_x, crouch_z], [rl_x, gz]]))
    # Lift RL
    phases.append(np.array([[fr_x, crouch_z], [fl_x, gz], [rr_x, crouch_z], [rl_x, lift_z]]))
    # Move RL forward
    phases.append(np.array([[fr_x, crouch_z], [fl_x, gz], [rr_x, crouch_z], [rl_x + d, lift_z]]))
    # Plant RL, restore others
    rl_x += d
    phases.append(np.array([[fr_x, gz], [fl_x, gz], [rr_x, gz], [rl_x, gz]]))

    # --- Move FL forward ---
    # FL is front-left. Shift weight to RIGHT side: bend FR and RR deeper.
    phases.append(np.array([[fr_x, crouch_z], [fl_x, gz], [rr_x, crouch_z], [rl_x, gz]]))
    # Lift FL
    phases.append(np.array([[fr_x, crouch_z], [fl_x, lift_z], [rr_x, crouch_z], [rl_x, gz]]))
    # Move FL forward
    phases.append(np.array([[fr_x, crouch_z], [fl_x + d, lift_z], [rr_x, crouch_z], [rl_x, gz]]))
    # Plant FL, restore others
    fl_x += d
    phases.append(np.array([[fr_x, gz], [fl_x, gz], [rr_x, gz], [rl_x, gz]]))

    # --- Move RR forward ---
    # RR is rear-right. Shift weight to LEFT side: bend FL and RL deeper.
    phases.append(np.array([[fr_x, gz], [fl_x, crouch_z], [rr_x, gz], [rl_x, crouch_z]]))
    # Lift RR
    phases.append(np.array([[fr_x, gz], [fl_x, crouch_z], [rr_x, lift_z], [rl_x, crouch_z]]))
    # Move RR forward
    phases.append(np.array([[fr_x, gz], [fl_x, crouch_z], [rr_x + d, lift_z], [rl_x, crouch_z]]))
    # Plant RR, restore others
    rr_x += d
    phases.append(np.array([[fr_x, gz], [fl_x, gz], [rr_x, gz], [rl_x, gz]]))

    return phases


def run_test(name, d, lift_z, crouch_z, phase_dur, cycles):
    print(f"\n{'='*60}")
    print(f"  {name}: d={d:+.3f}, lift={lift_z}, crouch={crouch_z}, dur={phase_dur}, cycles={cycles}")
    print(f"{'='*60}")

    base_env = Go2Env(
        xml_path="go2/scene.xml", control_substeps=4, kp=80.0, kd=4.0,
        initial_base_height=0.27,
        initial_angles=[0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
        settle_steps=500,
    )
    base_env.reset()
    kin = Go2Kinematics(base_env.model)

    phases_feet = make_sequential_gait(d, lift_z, crouch_z)
    joint_phases = [kin.policy_to_joints(f) for f in phases_feet]
    base_env.close()

    base_env2 = Go2Env(
        xml_path="go2/scene.xml", control_substeps=4, kp=80.0, kd=4.0,
        initial_base_height=0.27,
        initial_angles=[0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
        settle_steps=500,
    )
    base_env2.reset()
    env = RenderingEnv(base_env2)
    robot = RobotAPI(env)

    leg_order = ["FR", "RL", "FL", "RR"]
    step_names = ["shift", "lift", "move", "plant"]

    for cycle in range(cycles):
        for p_idx, phase in enumerate(joint_phases):
            robot.set_joints(phase)
            robot.step(phase_dur)
            state = robot.get_state()
            leg = leg_order[p_idx // 4]
            step = step_names[p_idx % 4]
            if cycle < 2:
                print(f"  C{cycle+1} {leg:2s} {step:5s}: x={state['x']:+.4f} h={state['h']:.4f}")

    state = robot.get_state()
    video_dir = Path("logs/test_walk_sequential")
    video_dir.mkdir(parents=True, exist_ok=True)
    env.save_video(video_dir / f"{name}.mp4")
    env.close()

    direction = "FORWARD" if state['x'] > 0.01 else "BACKWARD" if state['x'] < -0.01 else "NONE"
    print(f"\n  Final: x={state['x']:+.4f} h={state['h']:.4f} "
          f"roll={state['roll']:.4f} pitch={state['pitch']:.4f}  {direction}")
    return state['x']


def main():
    print("=" * 60)
    print("  Sequential Walk (one leg at a time, weight shift by bending)")
    print("=" * 60)

    results = {}
    # Positive delta
    results['plus_d03'] = run_test("plus_d03", d=+0.03, lift_z=-0.15, crouch_z=-0.30, phase_dur=0.5, cycles=3)
    results['plus_d04'] = run_test("plus_d04", d=+0.04, lift_z=-0.15, crouch_z=-0.30, phase_dur=0.5, cycles=3)

    # Negative delta
    results['minus_d03'] = run_test("minus_d03", d=-0.03, lift_z=-0.15, crouch_z=-0.30, phase_dur=0.5, cycles=3)
    results['minus_d04'] = run_test("minus_d04", d=-0.04, lift_z=-0.15, crouch_z=-0.30, phase_dur=0.5, cycles=3)

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for name, x in results.items():
        direction = "FORWARD" if x > 0.01 else "BACKWARD" if x < -0.01 else "NONE"
        print(f"  {name:15s}  x={x:+.4f}  {direction}")
    print(f"\n  Videos: logs/test_walk_sequential/")
    print(f"{'='*60}")


if __name__ == "__main__":
    sys.exit(main())