"""Compute joint angles from known-good foot positions using the existing IK pipeline.

Usage:
    python compute_primitives.py
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from env.env import Go2Env
from world.kinematics import Go2Kinematics


LEG_NAMES = ["FR", "FL", "RR", "RL"]
JOINT_NAMES_PER_LEG = ["hip", "thigh", "calf"]


def print_joints(name, joints):
    """Pretty-print a 12-dim joint vector."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for leg_idx, leg in enumerate(LEG_NAMES):
        base = leg_idx * 3
        print(f"  {leg}:")
        for j, jname in enumerate(JOINT_NAMES_PER_LEG):
            print(f"    {jname:6s} = {joints[base + j]:+.6f}")
    print(f"\n  Raw array:")
    print(f"  {repr(joints)}")


def print_foot_targets(name, feet):
    """Pretty-print foot targets."""
    print(f"\n  Foot targets for {name}:")
    for leg_idx, leg in enumerate(LEG_NAMES):
        print(f"    {leg}: foot_x={feet[leg_idx, 0]:+.4f}, foot_z={feet[leg_idx, 1]:+.4f}")


def main():
    # Create env to get the MuJoCo model for kinematics
    env = Go2Env(
        xml_path="go2/scene.xml",
        control_substeps=4,
        kp=80.0,
        kd=4.0,
        initial_base_height=0.27,
        initial_angles=[0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
        settle_steps=500,
    )
    env.reset()
    kin = Go2Kinematics(env.model)

    # ---------------------------------------------------------------
    # STAND
    # ---------------------------------------------------------------
    stand_feet = np.array([
        [ 0.1934, -0.27],   # FR
        [ 0.1934, -0.27],   # FL
        [-0.1934, -0.27],   # RR
        [-0.1934, -0.27],   # RL
    ])
    print_foot_targets("STAND", stand_feet)
    print_joints("STAND", kin.policy_to_joints(stand_feet))

    # ---------------------------------------------------------------
    # LAY
    # ---------------------------------------------------------------
    lay_feet = np.array([
        [ 0.1934, -0.093],  # FR
        [ 0.1934, -0.093],  # FL
        [-0.1934, -0.093],  # RR
        [-0.1934, -0.093],  # RL
    ])
    print_foot_targets("LAY", lay_feet)
    print_joints("LAY", kin.policy_to_joints(lay_feet))

    # ---------------------------------------------------------------
    # SIT
    # ---------------------------------------------------------------
    sit_feet = np.array([
        [ 0.1934, -0.27],   # FR
        [ 0.1934, -0.27],   # FL
        [-0.02,   -0.02],   # RR
        [-0.02,   -0.02],   # RL
    ])
    print_foot_targets("SIT", sit_feet)
    print_joints("SIT", kin.policy_to_joints(sit_feet))

    # ---------------------------------------------------------------
    # WALK — Phase 1
    # Pair A (FL+RR) swings forward, Pair B (FR+RL) planted
    # ---------------------------------------------------------------
    walk_phase1_feet = np.array([
        [ 0.1534, -0.27],   # FR — planted
        [ 0.2334, -0.15],   # FL — swinging
        [-0.1534, -0.15],   # RR — swinging
        [-0.2334, -0.27],   # RL — planted
    ])
    print_foot_targets("WALK PHASE 1", walk_phase1_feet)
    print_joints("WALK PHASE 1", kin.policy_to_joints(walk_phase1_feet))

    # ---------------------------------------------------------------
    # WALK — Phase 2
    # All legs planted (transition)
    # ---------------------------------------------------------------
    walk_phase2_feet = np.array([
        [ 0.1334, -0.27],   # FR
        [ 0.2334, -0.27],   # FL
        [-0.1534, -0.27],   # RR
        [-0.2534, -0.27],   # RL
    ])
    print_foot_targets("WALK PHASE 2", walk_phase2_feet)
    print_joints("WALK PHASE 2", kin.policy_to_joints(walk_phase2_feet))

    # ---------------------------------------------------------------
    # WALK — Phase 3
    # Pair B (FR+RL) swings forward, Pair A (FL+RR) planted
    # ---------------------------------------------------------------
    walk_phase3_feet = np.array([
        [ 0.2334, -0.15],   # FR — swinging
        [ 0.2134, -0.27],   # FL — planted
        [-0.1734, -0.27],   # RR — planted
        [-0.1534, -0.15],   # RL — swinging
    ])
    print_foot_targets("WALK PHASE 3", walk_phase3_feet)
    print_joints("WALK PHASE 3", kin.policy_to_joints(walk_phase3_feet))

    # ---------------------------------------------------------------
    # WALK — Phase 4
    # All legs planted (transition)
    # ---------------------------------------------------------------
    walk_phase4_feet = np.array([
        [ 0.2334, -0.27],   # FR
        [ 0.1934, -0.27],   # FL
        [-0.1934, -0.27],   # RR
        [-0.1534, -0.27],   # RL
    ])
    print_foot_targets("WALK PHASE 4", walk_phase4_feet)
    print_joints("WALK PHASE 4", kin.policy_to_joints(walk_phase4_feet))

    env.close()
    print(f"\n{'='*60}")
    print("  Done. All joint angles computed from existing IK pipeline.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()