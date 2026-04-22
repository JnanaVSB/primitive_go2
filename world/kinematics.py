"""2-link planar inverse kinematics for the Go2 quadruped.

For sit/stand/lay, each leg moves only in its sagittal plane. Hip abduction
stays at zero and foot_y is fixed by morphology. This reduces IK to a 2-link
planar problem (thigh + calf) per leg, with closed-form solution.

Geometry is extracted from the MuJoCo model at init time.

Action-space convention:
    LLM outputs per-leg (foot_x, foot_z) in BODY frame (base_link origin).
    This module converts to per-leg THIGH frame (the root of each 2-link chain)
    and solves IK. Returns 12-dim joint vector in JOINT_NAMES order.

Joint angle conventions (from go2.xml):
    hip (abduction): 0 (fixed for these tasks; range [-1.047, 1.047])
    thigh:           positive = rotates leg forward/down (range [-1.57, 3.49])
    calf:            negative = knee folded (range [-2.72, -0.84])

Go2 has L1 == L2 == 0.213 m (thigh length == calf length), which means max
reach is 0.426 m and the leg can fold to touch its own thigh origin.

Unreachable targets are clamped to the reachable envelope. A warning is logged.
"""

import logging
import numpy as np
import mujoco

logger = logging.getLogger(__name__)


# Must match env.py's JOINT_NAMES ordering (actuator order).
LEG_NAMES = ["FR", "FL", "RR", "RL"]
JOINTS_PER_LEG = 3
NUM_JOINTS = 12


class Go2Kinematics:
    """Per-leg 2-link planar IK for the Go2.

    Reads leg segment lengths and thigh-joint positions from the MuJoCo model
    once at init. Subsequent IK calls are pure math.
    """

    def __init__(self, model: mujoco.MjModel):
        self._extract_geometry(model)

    def _extract_geometry(self, model: mujoco.MjModel):
        """Read leg geometry from the MuJoCo model.

        Sets:
            thigh_pos[leg]: (x, y, z) of {LEG}_thigh body in base_link frame.
                            This is the pivot point for the 2-link IK chain.
            L1:             thigh length (calf body's z-offset from thigh body)
            L2:             calf length (foot body's z-offset from calf body)
            max_reach:      L1 + L2
            min_reach:      |L1 - L2| (zero for Go2 since L1 == L2)
        """
        self.thigh_pos = {}
        for leg in LEG_NAMES:
            hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip")
            thigh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_thigh")
            # thigh body_pos is relative to its parent (hip). Hip body_pos is
            # relative to its parent (base_link). Compose them.
            hip_in_base = model.body_pos[hip_id]
            thigh_in_hip = model.body_pos[thigh_id]
            self.thigh_pos[leg] = (hip_in_base + thigh_in_hip).copy()

        # L1 from FR_calf's z-offset from FR_thigh
        calf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FR_calf")
        self.L1 = float(abs(model.body_pos[calf_id][2]))

        # L2 from FR_foot's z-offset from FR_calf
        foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "FR_foot")
        self.L2 = float(abs(model.body_pos[foot_id][2]))

        self.max_reach = self.L1 + self.L2
        self.min_reach = abs(self.L1 - self.L2)

        logger.info(
            f"Go2 geometry: L1={self.L1:.4f}m, L2={self.L2:.4f}m, "
            f"max_reach={self.max_reach:.4f}m, min_reach={self.min_reach:.4f}m"
        )
        for leg in LEG_NAMES:
            logger.info(f"  {leg}_thigh pivot in base frame: {self.thigh_pos[leg]}")

    def _solve_leg_ik(self, x: float, z: float) -> tuple[float, float]:
        """Planar 2-link IK in one leg's thigh-pivot frame.

        Given foot target (x, z) relative to the thigh joint, compute
        (thigh_angle, calf_angle) matching the Go2 URDF conventions:
            thigh positive = leg rotates forward/down from vertical
            calf  negative = knee folds (from straight)

        Assumes hip abduction = 0 so the leg lives in the x-z plane.
        Clamps unreachable targets.
        """
        d_sq = x * x + z * z
        d = float(np.sqrt(d_sq))

        # Clamp beyond max reach
        if d > self.max_reach:
            scale = (self.max_reach - 1e-4) / d
            logger.warning(
                f"Foot target ({x:.3f}, {z:.3f}) exceeds max reach "
                f"{self.max_reach:.3f}; clamping"
            )
            x *= scale
            z *= scale
            d = self.max_reach - 1e-4
            d_sq = d * d

        # Clamp inside min reach (foot too close; leg would collide with itself)
        if d < max(self.min_reach, 1e-4):
            safe = max(self.min_reach, 1e-4) + 1e-4
            if d < 1e-6:
                # foot essentially at hip; pick a safe downward direction
                x, z = 0.0, -safe
            else:
                scale = safe / d
                x *= scale
                z *= scale
            logger.warning(
                f"Foot target below min reach {self.min_reach:.3f}; clamping"
            )
            d = safe
            d_sq = d * d

        # Knee angle from law of cosines.
        # Interior angle at the knee: cos(k) = (L1^2 + L2^2 - d^2) / (2 L1 L2)
        # Go2 calf is NEGATIVE when knee is bent. A straight leg is calf=0
        # (interior angle = pi). A folded leg is calf ≈ -pi (interior ≈ 0).
        cos_k = (self.L1**2 + self.L2**2 - d_sq) / (2 * self.L1 * self.L2)
        cos_k = float(np.clip(cos_k, -1.0, 1.0))
        k_interior = float(np.arccos(cos_k))  # 0..pi
        calf = -(np.pi - k_interior)  # 0 when straight, -pi when fully folded

        # Thigh angle.
        # Thigh=0 means leg points straight down (along -z), matching URDF.
        # At thigh=0, calf=0: foot at (0, -L1-L2), i.e. (0, -max_reach).
        # Angle from straight-down (-z axis) to foot direction:
        phi_to_foot = float(np.arctan2(x, -z))  # positive when foot is forward (+x)
        # Angle between thigh segment and the foot-direction line, at the hip:
        offset = float(np.arctan2(
            self.L2 * np.sin(k_interior),
            self.L1 - self.L2 * np.cos(k_interior),
        ))
        thigh = phi_to_foot + offset

        return float(thigh), float(calf)

    def policy_to_joints(self, foot_targets: np.ndarray) -> np.ndarray:
        """Convert body-frame foot targets to a 12-dim joint vector.

        Args:
            foot_targets: shape (4, 2). Row order matches LEG_NAMES.
                          Each row is (foot_x, foot_z) in base_link frame.

        Returns:
            12-dim joint vector in JOINT_NAMES order (FR_hip, FR_thigh, FR_calf,
            FL_hip, ..., RL_calf). Hip abduction entries are always 0.
        """
        foot_targets = np.asarray(foot_targets, dtype=np.float64)
        assert foot_targets.shape == (4, 2), (
            f"Expected foot_targets shape (4, 2), got {foot_targets.shape}"
        )

        joints = np.zeros(NUM_JOINTS, dtype=np.float64)
        for i, leg in enumerate(LEG_NAMES):
            foot_x_body, foot_z_body = foot_targets[i]
            thigh_x, _, thigh_z = self.thigh_pos[leg]

            # Base frame -> thigh-pivot frame (subtract thigh joint's position)
            x = foot_x_body - thigh_x
            z = foot_z_body - thigh_z

            thigh_angle, calf_angle = self._solve_leg_ik(x, z)

            base = i * JOINTS_PER_LEG
            joints[base + 0] = 0.0  # hip abduction
            joints[base + 1] = thigh_angle
            joints[base + 2] = calf_angle

        return joints

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """Compute foot positions in base frame from a 12-dim joint vector.

        Used for testing (round-trip) and for reading current foot state.
        Assumes hip abduction = 0.

        Args:
            joints: 12-dim joint vector in JOINT_NAMES order.

        Returns:
            Shape (4, 2). Row i is (foot_x, foot_z) in base_link frame for LEG_NAMES[i].
        """
        joints = np.asarray(joints, dtype=np.float64)
        assert joints.shape == (NUM_JOINTS,)

        foot_positions = np.zeros((4, 2), dtype=np.float64)
        for i, leg in enumerate(LEG_NAMES):
            base = i * JOINTS_PER_LEG
            thigh = joints[base + 1]
            calf = joints[base + 2]

            # Position of knee relative to thigh joint:
            #   thigh=0 means thigh points straight down (along -z).
            #   thigh>0 rotates leg forward (+x direction).
            knee_x = self.L1 * np.sin(thigh)
            knee_z = -self.L1 * np.cos(thigh)

            # Position of foot relative to knee:
            #   calf angle is relative to the thigh segment.
            #   At calf=0, calf segment continues along the thigh direction.
            total_angle = thigh + calf
            foot_x_from_knee = self.L2 * np.sin(total_angle)
            foot_z_from_knee = -self.L2 * np.cos(total_angle)

            # Foot in thigh-pivot frame
            x = knee_x + foot_x_from_knee
            z = knee_z + foot_z_from_knee

            # Thigh-pivot frame -> base frame
            thigh_x, _, thigh_z = self.thigh_pos[leg]
            foot_positions[i, 0] = x + thigh_x
            foot_positions[i, 1] = z + thigh_z

        return foot_positions