"""Tests for world/kinematics.py.

Validates the IK against ground-truth from the Go2 home keyframe and checks
forward-inverse round-trip consistency.
"""

import numpy as np
import mujoco
import pytest

from world.kinematics import Go2Kinematics, LEG_NAMES, NUM_JOINTS

# Adjust if your project layout puts the XML elsewhere
XML_PATH = "go2/go2.xml"


@pytest.fixture(scope="module")
def kin():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    return Go2Kinematics(model)


def test_geometry_extraction(kin):
    """Go2's thigh and calf are both 0.213 m."""
    assert abs(kin.L1 - 0.213) < 1e-4
    assert abs(kin.L2 - 0.213) < 1e-4
    assert abs(kin.max_reach - 0.426) < 1e-4


def test_thigh_pivot_symmetry(kin):
    """Front/rear thigh pivots should mirror in x; left/right should mirror in y."""
    fl = kin.thigh_pos["FL"]
    fr = kin.thigh_pos["FR"]
    rl = kin.thigh_pos["RL"]
    rr = kin.thigh_pos["RR"]

    # Front-back mirror in x
    assert abs(fl[0] - (-rl[0])) < 1e-6
    assert abs(fr[0] - (-rr[0])) < 1e-6
    # Left-right mirror in y
    assert abs(fl[1] - (-fr[1])) < 1e-6
    assert abs(rl[1] - (-rr[1])) < 1e-6
    # All at z=0 (hips mounted at body level)
    assert abs(fl[2]) < 1e-6


def test_home_keyframe_forward_kinematics(kin):
    """Forward kinematics on home pose (thigh=0.9, calf=-1.8) should give
    sensible foot positions: feet below the body, symmetric.
    """
    # Home keyframe: all legs thigh=0.9, calf=-1.8, hip=0
    joints = np.zeros(NUM_JOINTS)
    for i in range(4):
        joints[i * 3 + 0] = 0.0   # hip
        joints[i * 3 + 1] = 0.9   # thigh
        joints[i * 3 + 2] = -1.8  # calf

    feet = kin.forward_kinematics(joints)
    assert feet.shape == (4, 2)

    # All feet should be below the body origin (z < 0)
    for i, leg in enumerate(LEG_NAMES):
        assert feet[i, 1] < 0, f"{leg} foot should be below body, got z={feet[i, 1]:.3f}"

    # Foot z should be consistent across all four legs (same joint angles)
    z_values = feet[:, 1]
    assert np.std(z_values) < 1e-6, f"Foot z's differ: {z_values}"

    # Foot x should match thigh pivot x for all legs (straight down under thigh)
    # Actually not exactly — with thigh=0.9, the leg rotates forward, so feet
    # are forward of their thigh pivots.
    # Just check: front feet have foot_x > rear feet foot_x
    fr_x = feet[LEG_NAMES.index("FR"), 0]
    rr_x = feet[LEG_NAMES.index("RR"), 0]
    assert fr_x > rr_x, f"Front foot x ({fr_x:.3f}) should exceed rear ({rr_x:.3f})"


def test_round_trip_home(kin):
    """FK(home) -> IK -> joints should recover original thigh/calf angles."""
    joints_in = np.zeros(NUM_JOINTS)
    for i in range(4):
        joints_in[i * 3 + 1] = 0.9
        joints_in[i * 3 + 2] = -1.8

    feet = kin.forward_kinematics(joints_in)
    joints_out = kin.policy_to_joints(feet)

    np.testing.assert_allclose(joints_out, joints_in, atol=1e-6)


def test_round_trip_various_poses(kin):
    """Round-trip over a grid of thigh/calf angles within joint limits."""
    # Thigh: front range [-1.57, 3.49], back range [-0.52, 4.54]
    # Calf:  [-2.72, -0.84]
    # Pick angles safely inside all ranges.
    test_thighs = [0.3, 0.9, 1.5, 2.0]
    test_calfs = [-2.5, -2.0, -1.5, -1.0]

    for t in test_thighs:
        for c in test_calfs:
            joints_in = np.zeros(NUM_JOINTS)
            for i in range(4):
                joints_in[i * 3 + 1] = t
                joints_in[i * 3 + 2] = c

            feet = kin.forward_kinematics(joints_in)
            joints_out = kin.policy_to_joints(feet)

            np.testing.assert_allclose(
                joints_out, joints_in, atol=1e-6,
                err_msg=f"Round-trip failed at thigh={t}, calf={c}"
            )


def test_unreachable_target_clamps(kin, caplog):
    """Foot target beyond max reach should be clamped (not NaN) and log a warning."""
    import logging
    caplog.set_level(logging.WARNING, logger="world.kinematics")

    # All four feet at x=1.0, z=-1.0 (distance 1.41, way beyond 0.426 reach)
    feet = np.tile([1.0, -1.0], (4, 1))
    joints = kin.policy_to_joints(feet)

    assert not np.any(np.isnan(joints)), "IK should not return NaN"
    assert np.all(np.isfinite(joints)), "IK should return finite values"
    assert any("exceeds max reach" in rec.message for rec in caplog.records), \
        "Expected a clamping warning"


def test_hip_abduction_always_zero(kin):
    """policy_to_joints should set hip abduction to 0 regardless of input."""
    feet = np.array([
        [0.19, -0.27],   # FR (approximately home-pose foot position)
        [0.19, -0.27],   # FL
        [-0.19, -0.27],  # RR
        [-0.19, -0.27],  # RL
    ])
    joints = kin.policy_to_joints(feet)

    for i in range(4):
        assert joints[i * 3] == 0.0, f"Hip abduction for leg {i} should be 0"