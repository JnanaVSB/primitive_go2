"""Walk gait for the Go2 — parameterized Bezier planner.

The walk primitive is the planner itself, not a frozen keyframe list.
The LLM (or robot API) constructs a planner with desired parameters
(stride, swing height, body height) and queries it once per control
step:

    planner = make_walk_planner(stride=0.10, swing_height=0.08)
    phi = (elapsed_time / cycle_period) % 1.0
    joints = walk_step(planner, phi, kinematics)
    robot.set_joints(joints)

The planner is a 12-control-point Bezier swing trajectory + cosine-
blended stance, following the MIT Cheetah leg-trajectory planner
(Bledt et al. 2018; Zhang et al. 2019).

Run this module directly to inspect the gait in the MuJoCo viewer:

    python -m world.walk_gait
    python -m world.walk_gait --cycle-period 0.6
    python -m world.walk_gait --stride 0.06 --swing-height 0.05

Future: 3D extension for sidewalk and turning will add a `y` component
to the foot trajectory and require 3D IK. Not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
import numpy as np


# --------------------------------------------------------------------------
# Gait parameters
# --------------------------------------------------------------------------

@dataclass
class GaitParams:
    """Tunable parameters for the Bezier trot.

    Direction (forward/backward) is NOT a parameter. The planner produces
    one canonical trajectory; reverse direction is achieved at the runner
    level by stepping phi in reverse.
    """
    stride_length: float = 0.10        # m — total forward foot travel per cycle
    swing_height: float = 0.08         # m — peak foot lift during swing
    body_height: float = 0.27          # m — nominal foot_z when planted
    stance_penetration: float = 0.005  # m — small ground push for tracking
    stance_fraction: float = 0.5       # fraction of cycle in stance per leg
    # Phase offsets per leg in LEG_NAMES order (FR, FL, RR, RL).
    # Trot = diagonal pairs synchronized: FL+RR vs FR+RL.
    leg_phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)


# --------------------------------------------------------------------------
# Bezier gait planner
# --------------------------------------------------------------------------

# Home foot_x for each leg in body frame (matches hip-x in the Go2 model).
# LEG_NAMES order: FR, FL, RR, RL.
HOME_FOOT_X = np.array([0.1934, 0.1934, -0.1934, -0.1934])


class BezierGaitPlanner:
    """Per-leg foot trajectories for a trot gait.

    Each leg's foot follows a closed cycle in (foot_x, foot_z) body-frame
    coordinates with two parts:

      Stance (phi_leg < stance_fraction):
        Foot on the ground, sliding from +stride/2 to -stride/2 with a
        cosine blend. Tiny ground penetration keeps the PD controller
        engaged with the floor.

      Swing (phi_leg >= stance_fraction):
        Foot lifts and arcs forward. 12-control-point Bezier curve
        gives continuous velocity at lift-off and touchdown plus a
        smooth foot-clearance arc.

    Trot synchronization is by phase offset: diagonal pairs share an
    offset of 0; the other diagonal has 0.5.
    """

    # 12 control points for the swing-phase Bezier curve, normalized to
    # unit stride (x in [-0.5, 0.5]) and unit swing height (z in [0, 1]).
    # MIT Cheetah trajectory shape (Zhang et al. 2019, Table II).
    _SWING_CONTROL_POINTS = np.array([
        [-0.50, 0.00],
        [-0.65, 0.00],
        [-0.65, 0.90],
        [-0.65, 0.90],
        [-0.65, 0.90],
        [ 0.00, 0.90],
        [ 0.00, 0.90],
        [ 0.00, 1.20],
        [ 0.65, 1.20],
        [ 0.65, 1.20],
        [ 0.50, 0.00],
        [ 0.50, 0.00],
    ])

    def __init__(self, params: GaitParams | None = None):
        self.params = params or GaitParams()

    def foot_targets(self, phi: float) -> np.ndarray:
        """Foot positions for all 4 legs at cycle phase phi ∈ [0, 1).

        Returns a (4, 2) array of (foot_x, foot_z) in body frame, ready
        for Go2Kinematics.policy_to_joints().
        """
        # Reverse the cycle direction so the canonical trajectory walks
        # forward in our env. The Bezier swing curve has an asymmetric
        # shape (designed for one direction of x sweep); reversing phi
        # preserves that shape while flipping the gait direction. This
        # is the empirical fix — convention sign-flips in _stance/_swing
        # individually distort the swing arc.
        phi = (1.0 - phi) % 1.0

        targets = np.zeros((4, 2))
        for leg_idx in range(4):
            offset = self.params.leg_phase_offsets[leg_idx]
            phi_leg = (phi + offset) % 1.0

            if phi_leg < self.params.stance_fraction:
                phi_st = phi_leg / self.params.stance_fraction
                fx_rel, fz = self._stance(phi_st)
            else:
                phi_sw = (phi_leg - self.params.stance_fraction) / (
                    1.0 - self.params.stance_fraction
                )
                fx_rel, fz = self._swing(phi_sw)

            targets[leg_idx, 0] = HOME_FOOT_X[leg_idx] + fx_rel
            targets[leg_idx, 1] = fz
        return targets

    def _stance(self, phi_st: float) -> tuple[float, float]:
        """Stance phase: foot slides backward in body frame. phi_st ∈ [0, 1)."""
        # Cosine blend: foot at +stride/2 at phi_st=0, -stride/2 at phi_st=1.
        s = 0.5 * (1.0 + np.cos(np.pi * phi_st))   # 1 -> 0
        foot_x = self.params.stride_length * (s - 0.5)
        # Push down at mid-stance, zero at endpoints.
        push = self.params.stance_penetration * np.sin(np.pi * phi_st)
        foot_z = -self.params.body_height - push
        return float(foot_x), float(foot_z)

    def _swing(self, phi_sw: float) -> tuple[float, float]:
        """Swing phase: foot arcs forward through the air. phi_sw ∈ [0, 1)."""
        n = len(self._SWING_CONTROL_POINTS) - 1
        point = np.zeros(2)
        for k, pk in enumerate(self._SWING_CONTROL_POINTS):
            coeff = comb(n, k) * (phi_sw**k) * ((1.0 - phi_sw) ** (n - k))
            point += coeff * pk
        foot_x = self.params.stride_length * point[0]
        foot_z = -self.params.body_height + self.params.swing_height * point[1]
        return float(foot_x), float(foot_z)


# --------------------------------------------------------------------------
# LLM-facing API — thin wrappers used by the runner / robot API.
# --------------------------------------------------------------------------

def make_walk_planner(
    stride_length: float = 0.10,
    swing_height: float = 0.08,
    body_height: float = 0.27,
    stance_fraction: float = 0.5,
) -> BezierGaitPlanner:
    """Construct a BezierGaitPlanner with the given gait parameters.

    The robot API (or LLM, indirectly) calls this once per walk command.
    All parameters have safe Go2 defaults.
    """
    return BezierGaitPlanner(GaitParams(
        stride_length=stride_length,
        swing_height=swing_height,
        body_height=body_height,
        stance_fraction=stance_fraction,
    ))


def walk_step(
    planner: BezierGaitPlanner,
    phi: float,
    kinematics,
) -> np.ndarray:
    """Compute joint angles at cycle phase phi.

    Args:
        planner:    a BezierGaitPlanner.
        phi:        cycle phase in [0, 1).
        kinematics: a Go2Kinematics instance.

    Returns:
        12-dim numpy array of joint angles in JOINT_NAMES order.
    """
    foot_targets = planner.foot_targets(phi)
    return kinematics.policy_to_joints(foot_targets)


# --------------------------------------------------------------------------
# Viewer — drives the planner inside Go2Env at every control step.
# Used to inspect the gait visually; not on the runtime path.
# --------------------------------------------------------------------------

def _run_in_viewer(
    planner: BezierGaitPlanner,
    cycle_period: float = 0.4,
    total_duration: float = 10.0,
    xml_path: str = "go2/scene.xml",
    kp: float = 80.0,
    kd: float = 4.0,
) -> None:
    """Run the planner inside Go2Env, visualize in the MuJoCo viewer.

    At each control step:
      1. Compute cycle phase phi from elapsed time.
      2. Ask planner for body-frame foot targets.
      3. IK to 12-dim joint vector.
      4. Send to env's PD controller.

    The env settles at phi=0 of the gait pose; one full cycle then runs
    silently before the viewer opens, so physics catches up before we
    start watching.
    """
    import time
    import mujoco
    import mujoco.viewer
    from env.env import Go2Env
    from world.kinematics import Go2Kinematics

    # Settle env at phi=0 of the gait pose so the first command isn't a
    # leap from home pose.
    initial_targets = planner.foot_targets(0.0)
    model_for_kin = mujoco.MjModel.from_xml_path(xml_path)
    kin = Go2Kinematics(model_for_kin)
    initial_joints = kin.policy_to_joints(initial_targets)

    env = Go2Env(
        xml_path=xml_path,
        control_substeps=4,
        kp=kp,
        kd=kd,
        initial_base_height=planner.params.body_height,
        initial_angles=list(initial_joints),
        settle_steps=500,
    )
    env.reset()

    # Rebind kinematics to the env's actual model for consistency.
    kin = Go2Kinematics(env.model)
    dt = env.model.opt.timestep * env.control_substeps

    print(f"\nOpening viewer.")
    print(f"  cycle_period={cycle_period}s, dt={dt*1000:.1f}ms per step")
    print(f"  total_duration={total_duration}s "
          f"(~{total_duration / cycle_period:.1f} cycles)")
    print(f"  GaitParams: stride={planner.params.stride_length}, "
          f"swing_height={planner.params.swing_height}, "
          f"body_height={planner.params.body_height}")
    print("  Close the window or press ESC to exit.")

    # Settle phase: run one full cycle silently before the viewer opens.
    settle_steps = int(cycle_period / dt)
    for k in range(settle_steps):
        phi = (k * dt / cycle_period) % 1.0
        env.step(walk_step(planner, phi, kin))

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        elapsed = 0.0
        while viewer.is_running() and elapsed < total_duration:
            phi = (elapsed / cycle_period) % 1.0
            env.step(walk_step(planner, phi, kin))
            viewer.sync()
            time.sleep(dt)
            elapsed += dt

        # Hold final pose so the user can inspect.
        final = env.data.qpos[env._qpos_idx].copy()
        while viewer.is_running():
            env.step(final)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize the Bezier walk gait in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--cycle-period", type=float, default=0.4,
        help="Seconds per full gait cycle (default: 0.4).",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Total seconds to walk (default: 10.0).",
    )
    parser.add_argument(
        "--stride", type=float, default=0.10,
        help="Stride length in m (default: 0.10).",
    )
    parser.add_argument(
        "--swing-height", type=float, default=0.08,
        help="Peak foot lift in m (default: 0.08).",
    )
    parser.add_argument(
        "--body-height", type=float, default=0.27,
        help="Body height in m (default: 0.27).",
    )
    parser.add_argument(
        "--kp", type=float, default=80.0,
        help="PD position gain (default: 80).",
    )
    parser.add_argument(
        "--kd", type=float, default=4.0,
        help="PD damping gain (default: 4).",
    )
    args = parser.parse_args()

    planner = make_walk_planner(
        stride_length=args.stride,
        swing_height=args.swing_height,
        body_height=args.body_height,
    )
    _run_in_viewer(
        planner,
        cycle_period=args.cycle_period,
        total_duration=args.duration,
        kp=args.kp,
        kd=args.kd,
    )

# """Walk gait for the Go2 — continuous Bezier planner.

# This module is run standalone to visualize a trot in the MuJoCo viewer:

#     python -m world.walk_gait
#     python -m world.walk_gait --cycle-period 0.5 --duration 20
#     python -m world.walk_gait --x-vel 0.3   # smaller stride

# The planner runs at every control step. Foot positions come from a
# 12-control-point Bezier swing trajectory + cosine-blended stance,
# following the MIT cheetah leg-trajectory planner. Each foot's body-frame
# target is converted to joint angles via Go2Kinematics and sent to the
# PD controller.

# No precomputed keyframes — the planner *is* the trajectory.

# Future: the LLM-facing API for walking will be added later. It will be
# either a time-parameterized callable (`get_walk_trajectory(duration) ->
# callable`) or a method on the robot API (`robot.walk_for(seconds)`).
# That decision is deferred to when we wire this into primitives.py.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from math import comb
# import numpy as np


# # --------------------------------------------------------------------------
# # Gait parameters
# # --------------------------------------------------------------------------

# @dataclass
# class GaitParams:
#     """Tunable parameters for the Bezier trot.

#     Conservative defaults: small stride, generous foot clearance, slow
#     cycle. Tune up once the trot is visibly stable.
#     """
#     stride_length: float = 0.10        # m — total forward foot travel per cycle
#     swing_height: float = 0.08         # m — peak foot lift during swing
#     body_height: float = 0.27          # m — nominal foot_z when planted
#     stance_penetration: float = 0.005  # m — small ground push for tracking
#     stance_fraction: float = 0.5       # fraction of cycle in stance per leg
#     # Phase offsets per leg in LEG_NAMES order (FR, FL, RR, RL).
#     # Trot = diagonal pairs synchronized: FL+RR vs FR+RL.
#     leg_phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)
#     # If True, body moves in +x direction (forward). Flip if it walks
#     # the wrong way in the viewer — purely a sign convention.
#     forward: bool = True


# # --------------------------------------------------------------------------
# # Bezier gait planner
# # --------------------------------------------------------------------------

# # Home foot_x for each leg in body frame (matches hip-x in shared/robot.j2).
# # LEG_NAMES order: FR, FL, RR, RL.
# HOME_FOOT_X = np.array([0.1934, 0.1934, -0.1934, -0.1934])


# class BezierGaitPlanner:
#     """Per-leg foot trajectories for a trot gait.

#     Each leg's foot follows a closed cycle in (foot_x, foot_z) body-frame
#     coordinates with two parts:

#       Stance (phi_leg < stance_fraction): foot on the ground, sliding
#         backward from +stride/2 to -stride/2 with a cosine blend. Tiny
#         ground penetration keeps the PD controller engaged.

#       Swing (phi_leg >= stance_fraction): foot lifts and arcs forward.
#         12-control-point Bezier curve (continuous velocity at lift-off
#         and touchdown, smooth foot-clearance arc).
#     """

#     # 12 control points for the swing-phase Bezier curve, normalized to
#     # unit stride (x in [-0.5, 0.5]) and unit swing height (z in [0, 1]).
#     # MIT cheetah trajectory shape from Zhang et al. 2019, Table II.
#     _SWING_CONTROL_POINTS = np.array([
#         [-0.50, 0.00],
#         [-0.65, 0.00],
#         [-0.65, 0.90],
#         [-0.65, 0.90],
#         [-0.65, 0.90],
#         [ 0.00, 0.90],
#         [ 0.00, 0.90],
#         [ 0.00, 1.20],
#         [ 0.65, 1.20],
#         [ 0.65, 1.20],
#         [ 0.50, 0.00],
#         [ 0.50, 0.00],
#     ])

#     def __init__(self, params: GaitParams | None = None):
#         self.params = params or GaitParams()

#     def foot_targets(self, phi: float) -> np.ndarray:
#         """Foot positions for all 4 legs at cycle phase phi ∈ [0, 1).

#         Returns a (4, 2) array of (foot_x, foot_z) in body frame, ready
#         to feed into Go2Kinematics.policy_to_joints().
#         """
#         targets = np.zeros((4, 2))
#         for leg_idx in range(4):
#             offset = self.params.leg_phase_offsets[leg_idx]
#             phi_leg = (phi + offset) % 1.0

#             if phi_leg < self.params.stance_fraction:
#                 phi_st = phi_leg / self.params.stance_fraction
#                 fx_rel, fz = self._stance(phi_st)
#             else:
#                 phi_sw = (phi_leg - self.params.stance_fraction) / (
#                     1.0 - self.params.stance_fraction
#                 )
#                 fx_rel, fz = self._swing(phi_sw)

#             # Apply forward/backward sign and add the leg's home offset.
#             if not self.params.forward:
#                 fx_rel = -fx_rel
#             targets[leg_idx, 0] = HOME_FOOT_X[leg_idx] + fx_rel
#             targets[leg_idx, 1] = fz
#         return targets

#     def _stance(self, phi_st: float) -> tuple[float, float]:
#         """Stance phase: foot slides backward in body frame. phi_st ∈ [0, 1)."""
#         # Cosine blend: foot at +stride/2 at phi_st=0, -stride/2 at phi_st=1.
#         s = 0.5 * (1.0 + np.cos(np.pi * phi_st))   # 1 -> 0
#         foot_x = self.params.stride_length * (s - 0.5)
#         # Push down at mid-stance, zero at endpoints.
#         push = self.params.stance_penetration * np.sin(np.pi * phi_st)
#         foot_z = -self.params.body_height - push
#         return float(foot_x), float(foot_z)

#     def _swing(self, phi_sw: float) -> tuple[float, float]:
#         """Swing phase: foot arcs forward through the air. phi_sw ∈ [0, 1)."""
#         n = len(self._SWING_CONTROL_POINTS) - 1
#         point = np.zeros(2)
#         for k, pk in enumerate(self._SWING_CONTROL_POINTS):
#             coeff = comb(n, k) * (phi_sw**k) * ((1.0 - phi_sw) ** (n - k))
#             point += coeff * pk
#         foot_x = self.params.stride_length * point[0]
#         foot_z = -self.params.body_height + self.params.swing_height * point[1]
#         return float(foot_x), float(foot_z)


# # --------------------------------------------------------------------------
# # Viewer — drives the planner inside Go2Env at every control step.
# # --------------------------------------------------------------------------

# def _run_in_viewer(
#     planner: BezierGaitPlanner,
#     cycle_period: float = 0.4,
#     total_duration: float = 10.0,
#     xml_path: str = "go2/scene.xml",
#     kp: float = 80.0,
#     kd: float = 4.0,
# ) -> None:
#     """Run the planner inside Go2Env, visualize in the MuJoCo viewer.

#     At each control step:
#       1. Compute cycle phase phi from elapsed time.
#       2. Ask planner for body-frame foot targets.
#       3. IK to 12-dim joint vector.
#       4. Send to env's PD controller.

#     No interpolation between keyframes — the planner produces a smooth
#     trajectory directly, sampled at the env's control rate.
#     """
#     import time
#     import mujoco
#     import mujoco.viewer
#     from env.env import Go2Env
#     from world.kinematics import Go2Kinematics

#     # Settle the env at phi=0 of the gait so the first command isn't a
#     # leap from home pose.
#     initial_targets = planner.foot_targets(0.0)
#     # Need the kinematics to convert; load model briefly to instantiate.
#     model_for_kin = mujoco.MjModel.from_xml_path(xml_path)
#     kin = Go2Kinematics(model_for_kin)
#     initial_joints = kin.policy_to_joints(initial_targets)

#     env = Go2Env(
#         xml_path=xml_path,
#         control_substeps=4,
#         kp=kp,
#         kd=kd,
#         initial_base_height=planner.params.body_height,
#         initial_angles=list(initial_joints),
#         settle_steps=500,
#     )
#     env.reset()

#     # Rebind kinematics to the env's actual model for consistency.
#     kin = Go2Kinematics(env.model)

#     dt = env.model.opt.timestep * env.control_substeps

#     print(f"\nOpening viewer.")
#     print(f"  cycle_period={cycle_period}s, dt={dt*1000:.1f}ms per step")
#     print(f"  total_duration={total_duration}s "
#           f"(~{total_duration / cycle_period:.1f} cycles)")
#     print(f"  GaitParams: stride={planner.params.stride_length}, "
#           f"swing_height={planner.params.swing_height}, "
#           f"body_height={planner.params.body_height}, "
#           f"forward={planner.params.forward}")
#     print("  Close the window or press ESC to exit.")

#     with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
#         elapsed = 0.0
#         while viewer.is_running() and elapsed < total_duration:
#             phi = (elapsed / cycle_period) % 1.0
#             foot_targets = planner.foot_targets(phi)
#             joint_targets = kin.policy_to_joints(foot_targets)
#             env.step(joint_targets)
#             viewer.sync()
#             time.sleep(dt)
#             elapsed += dt

#         # Hold final pose so the user can inspect.
#         final = env.data.qpos[env._qpos_idx].copy()
#         while viewer.is_running():
#             env.step(final)
#             viewer.sync()
#             time.sleep(dt)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Visualize the Bezier walk gait in the MuJoCo viewer.",
#     )
#     parser.add_argument(
#         "--cycle-period", type=float, default=0.4,
#         help="Seconds per full gait cycle (default: 0.4).",
#     )
#     parser.add_argument(
#         "--duration", type=float, default=10.0,
#         help="Total seconds to walk (default: 10.0).",
#     )
#     parser.add_argument(
#         "--stride", type=float, default=None,
#         help="Override GaitParams.stride_length (m).",
#     )
#     parser.add_argument(
#         "--swing-height", type=float, default=None,
#         help="Override GaitParams.swing_height (m).",
#     )
#     parser.add_argument(
#         "--backward", action="store_true",
#         help="Walk backward (flip the sign convention).",
#     )
#     parser.add_argument(
#         "--kp", type=float, default=80.0,
#         help="PD position gain (default: 80).",
#     )
#     parser.add_argument(
#         "--kd", type=float, default=4.0,
#         help="PD damping gain (default: 4).",
#     )
#     args = parser.parse_args()

#     params = GaitParams(forward=not args.backward)
#     if args.stride is not None:
#         params.stride_length = args.stride
#     if args.swing_height is not None:
#         params.swing_height = args.swing_height

#     planner = BezierGaitPlanner(params)
#     _run_in_viewer(
#         planner,
#         cycle_period=args.cycle_period,
#         total_duration=args.duration,
#         kp=args.kp,
#         kd=args.kd,
#     )


# """Walk gait primitive for the Go2.

# Exposes the LLM-facing primitive `get_walk_phases()`, which returns one
# trot cycle as 24 joint-angle keyframes. The runtime path is a constant
# lookup (WALK_PHASES) — no math runs when policy code calls it.

# The keyframes in WALK_PHASES were extracted from the convex-MPC trot
# controller in https://github.com/elijah-waichong-chan/go2-convex-mpc,
# sampled with error-greedy sampling so keyframes cluster near gait
# events (touchdown / lift-off) rather than being uniformly spaced in
# time. See `extract_walk_keyframes.py` (lives in that repo, not this
# one) for the extraction procedure.

# To regenerate WALK_PHASES:
#     1. Clone go2-convex-mpc and follow its install instructions.
#     2. Drop extract_walk_keyframes.py into its repo root.
#     3. Run `python extract_walk_keyframes.py` (optionally with --x-vel,
#        --gait-hz, --n-keyframes, --replay full|keyframes).
#     4. Paste the printed `WALK_PHASES = [...]` block over the constant
#        below.

# Run this file as a script to inspect the gait in the MuJoCo viewer:

#     python -m world.walk_gait                 # loop walk for 10s
#     python -m world.walk_gait --duration 20   # loop walk for 20s
#     python -m world.walk_gait --cycle-period 1.0   # faster cycle
# """

# from __future__ import annotations

# import numpy as np


# # --------------------------------------------------------------------------
# # LLM-facing primitive — the only thing called from policy code at runtime.
# # --------------------------------------------------------------------------

# # 24 keyframes, one full trot cycle. Phase i is at phi = i / 24.
# # Layout: 12 joint angles in JOINT_NAMES order
# #   FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
# #   RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf.
# #
# # Source: convex-MPC trot at x_vel=0.5 m/s, gait_hz=2.5, sampled with
# # uniform sampling. See module docstring for regeneration steps.
# WALK_PHASES: list[np.ndarray] = [
#     np.array([+0.0595, +0.9148, -1.8858, -0.0186, +1.2237, -1.7679, +0.0077, +1.1220, -1.6272, +0.0381, +0.8396, -1.7570]),  # phase 0
#     np.array([+0.0344, +0.9413, -1.8750, -0.0367, +1.2441, -1.7453, -0.0101, +1.1590, -1.6307, +0.0250, +0.8614, -1.7407]),  # phase 1
#     np.array([+0.0025, +0.9751, -1.8884, -0.0654, +1.2548, -1.7245, -0.0364, +1.1766, -1.6196, -0.0041, +0.8773, -1.7294]),  # phase 2
#     np.array([+0.0119, +1.0092, -1.8821, -0.0552, +1.2708, -1.6977, -0.0242, +1.1858, -1.5841, +0.0056, +0.9074, -1.7249]),  # phase 3
#     np.array([+0.0228, +1.0344, -1.8705, -0.0379, +1.2981, -1.6980, -0.0175, +1.2091, -1.5801, +0.0152, +0.9373, -1.7245]),  # phase 4
#     np.array([-0.0037, +1.0766, -1.8664, -0.0483, +1.3911, -1.8226, -0.0649, +1.3134, -1.7297, -0.0145, +0.9789, -1.7213]),  # phase 5
#     np.array([+0.0126, +1.0999, -1.8584, -0.0268, +1.4874, -2.0125, -0.0639, +1.4071, -1.9265, +0.0017, +1.0074, -1.7253]),  # phase 6
#     np.array([+0.0211, +1.1263, -1.8541, -0.0172, +1.5372, -2.2614, -0.0466, +1.4455, -2.1676, +0.0087, +1.0285, -1.7161]),  # phase 7
#     np.array([-0.0049, +1.1452, -1.8541, -0.0615, +1.4525, -2.3507, -0.0521, +1.3587, -2.2530, -0.0211, +1.0319, -1.6911]),  # phase 8
#     np.array([+0.0048, +1.1547, -1.8361, -0.0833, +1.2845, -2.3293, -0.0286, +1.1856, -2.2027, -0.0134, +1.0388, -1.6697]),  # phase 9
#     np.array([+0.0115, +1.1714, -1.8061, -0.0824, +1.0511, -2.1460, -0.0144, +0.9591, -1.9884, -0.0089, +1.0602, -1.6482]),  # phase 10
#     np.array([-0.0084, +1.1946, -1.7861, -0.0888, +0.9490, -1.9838, -0.0419, +0.8634, -1.8342, -0.0307, +1.0907, -1.6399]),  # phase 11
#     np.array([+0.0025, +1.2245, -1.7529, -0.0640, +0.9316, -1.8893, -0.0456, +0.8470, -1.7436, -0.0198, +1.1463, -1.6473]),  # phase 12
#     np.array([+0.0251, +1.2417, -1.7306, -0.0360, +0.9640, -1.8912, -0.0251, +0.8693, -1.7323, +0.0015, +1.1780, -1.6468]),  # phase 13
#     np.array([+0.0510, +1.2540, -1.7023, -0.0076, +1.0062, -1.9038, +0.0025, +0.8906, -1.7190, +0.0243, +1.1940, -1.6231]),  # phase 14
#     np.array([+0.0255, +1.2699, -1.6838, -0.0324, +1.0289, -1.8893, -0.0219, +0.9183, -1.7191, -0.0007, +1.2013, -1.5932]),  # phase 15
#     np.array([+0.0161, +1.3065, -1.7030, -0.0311, +1.0569, -1.8802, -0.0183, +0.9481, -1.7156, +0.0076, +1.2388, -1.6155]),  # phase 16
#     np.array([+0.0231, +1.4131, -1.8669, -0.0085, +1.0983, -1.8751, +0.0074, +0.9885, -1.7119, +0.0528, +1.3592, -1.8079]),  # phase 17
#     np.array([-0.0100, +1.5017, -2.0699, -0.0366, +1.1181, -1.8651, -0.0213, +1.0136, -1.7153, +0.0348, +1.4409, -2.0084]),  # phase 18
#     np.array([+0.0065, +1.5073, -2.2903, -0.0240, +1.1457, -1.8645, -0.0052, +1.0277, -1.6957, +0.0306, +1.4393, -2.2236]),  # phase 19
#     np.array([+0.0568, +1.3894, -2.3432, -0.0050, +1.1617, -1.8584, +0.0174, +1.0299, -1.6681, +0.0348, +1.3154, -2.2661]),  # phase 20
#     np.array([+0.0692, +1.1482, -2.2419, -0.0315, +1.1710, -1.8251, -0.0070, +1.0420, -1.6409, +0.0014, +1.0654, -2.1147]),  # phase 21
#     np.array([+0.0776, +0.9975, -2.0747, -0.0187, +1.1887, -1.8043, +0.0082, +1.0619, -1.6244, +0.0177, +0.9196, -1.9380]),  # phase 22
#     np.array([+0.0736, +0.9146, -1.9020, -0.0090, +1.2185, -1.7733, +0.0194, +1.1079, -1.6199, +0.0449, +0.8396, -1.7735]),  # phase 23
# ]


# def get_walk_phases() -> list[np.ndarray]:
#     """Return one trot cycle as 24 joint-angle keyframes.

#     Each entry is a (12,) numpy array in JOINT_NAMES order. Robot API
#     consumes them sequentially with interpolation between, then loops
#     the cycle for the desired duration.

#     Returns defensive copies so callers can't mutate the constant.
#     """
#     return [phase.copy() for phase in WALK_PHASES]


# # --------------------------------------------------------------------------
# # Viewer — visualize the gait in MuJoCo. Not used at runtime.
# # --------------------------------------------------------------------------

# def _run_in_viewer(
#     joint_phases: list[np.ndarray],
#     cycle_period: float = 0.4,
#     total_duration: float = 10.0,
#     xml_path: str = "go2/scene.xml",
#     kp: float = 80.0,
#     kd: float = 4.0,
# ) -> None:
#     """Loop the keyframes in the MuJoCo viewer for visual sanity check.

#     Uses the same quintic interpolation as the runner, so the viewer
#     behavior matches what the FORGE loop will execute.

#     Default cycle_period=0.4s matches the source MPC's gait period
#     (1/2.5 Hz). Lower it for a quicker trot, raise it for slower.
#     """
#     import time
#     import mujoco
#     import mujoco.viewer
#     from env.env import Go2Env
#     from world.trajectory import make_trajectory, trajectory_duration_to_nsteps

#     # Initialize the env at phase 0 of the gait, not at home.
#     # Otherwise the first interpolation segment has to leap from home to
#     # phase 0 in cycle_period/n_phases seconds — usually too fast for
#     # the PD controller to track, and the robot collapses.
#     initial_angles = list(joint_phases[0])
#     env = Go2Env(
#         xml_path=xml_path,
#         control_substeps=4,
#         kp=kp,
#         kd=kd,
#         initial_base_height=0.27,
#         initial_angles=initial_angles,
#         settle_steps=500,
#     )
#     env.reset()

#     n_phases = len(joint_phases)
#     phase_duration = cycle_period / n_phases
#     dt = env.model.opt.timestep * env.control_substeps
#     n_steps_per_phase = trajectory_duration_to_nsteps(phase_duration, dt)

#     print(f"\nOpening viewer.")
#     print(f"  cycle_period={cycle_period}s, {n_phases} phases, "
#           f"{phase_duration:.3f}s each")
#     print(f"  total_duration={total_duration}s "
#           f"(~{total_duration / cycle_period:.1f} cycles)")
#     print("  Close the window or press ESC to exit.")

#     with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
#         elapsed = 0.0
#         phase_idx = 0
#         while viewer.is_running() and elapsed < total_duration:
#             start_joints = env.data.qpos[env._qpos_idx].copy()
#             target_joints = joint_phases[phase_idx]
#             traj = make_trajectory(start_joints, target_joints, phase_duration)

#             for step in range(n_steps_per_phase):
#                 if not viewer.is_running() or elapsed >= total_duration:
#                     break
#                 t = step * dt
#                 env.step(traj(t))
#                 viewer.sync()
#                 time.sleep(dt)
#                 elapsed += dt

#             phase_idx = (phase_idx + 1) % n_phases

#         # Hold the last commanded target so the user can inspect.
#         final = env.data.qpos[env._qpos_idx].copy()
#         while viewer.is_running():
#             env.step(final)
#             viewer.sync()
#             time.sleep(dt)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Visualize the walk gait keyframes in the MuJoCo viewer.",
#     )
#     parser.add_argument(
#         "--cycle-period", type=float, default=0.4,
#         help="Seconds per full gait cycle (default: 0.4, matches source MPC).",
#     )
#     parser.add_argument(
#         "--duration", type=float, default=10.0,
#         help="Total seconds to walk before holding (default: 10.0).",
#     )
#     args = parser.parse_args()

#     _run_in_viewer(
#         get_walk_phases(),
#         cycle_period=args.cycle_period,
#         total_duration=args.duration,
#     )