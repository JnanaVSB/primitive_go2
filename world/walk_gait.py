"""Walk gait for the Go2."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
import numpy as np


# Order:
# FR_hip, FR_thigh, FR_calf,
# FL_hip, FL_thigh, FL_calf,
# RR_hip, RR_thigh, RR_calf,
# RL_hip, RL_thigh, RL_calf

SAMPLED_JOINT_ANGLES: list[np.ndarray] = [
    np.array([-0.0100, +1.0732, -1.8219, +0.0039, +0.6710, -1.8133, -0.0048, +0.6782, -1.8076, +0.0111, +1.0678, -1.8226]),  # sample 0
    np.array([-0.0100, +1.0732, -1.8220, +0.0039, +0.6710, -1.8137, -0.0048, +0.6781, -1.8078, +0.0111, +1.0677, -1.8226]),  # sample 1
    np.array([-0.0103, +1.0736, -1.8230, +0.0039, +0.6716, -1.8159, -0.0047, +0.6786, -1.8096, +0.0114, +1.0674, -1.8230]),  # sample 2
    np.array([-0.0109, +1.0746, -1.8250, +0.0036, +0.6743, -1.8252, -0.0039, +0.6809, -1.8193, +0.0121, +1.0669, -1.8238]),  # sample 3
    np.array([-0.0119, +1.0759, -1.8280, +0.0032, +0.6794, -1.8423, -0.0028, +0.6852, -1.8370, +0.0133, +1.0657, -1.8247]),  # sample 4
    np.array([-0.0131, +1.0773, -1.8317, +0.0027, +0.6864, -1.8652, -0.0017, +0.6914, -1.8604, +0.0148, +1.0637, -1.8256]),  # sample 5
    np.array([-0.0145, +1.0784, -1.8358, +0.0024, +0.6956, -1.8918, -0.0008, +0.6998, -1.8874, +0.0168, +1.0607, -1.8263]),  # sample 6
    np.array([-0.0161, +1.0792, -1.8402, +0.0021, +0.7074, -1.9201, +0.0001, +0.7109, -1.9161, +0.0192, +1.0566, -1.8267]),  # sample 7
    np.array([-0.0178, +1.0794, -1.8447, +0.0018, +0.7225, -1.9489, +0.0008, +0.7253, -1.9452, +0.0220, +1.0515, -1.8268]),  # sample 8
    np.array([-0.0197, +1.0789, -1.8492, +0.0016, +0.7412, -1.9774, +0.0015, +0.7435, -1.9739, +0.0252, +1.0451, -1.8265]),  # sample 9
    np.array([-0.0217, +1.0774, -1.8537, +0.0014, +0.7640, -2.0049, +0.0021, +0.7656, -2.0016, +0.0286, +1.0375, -1.8257]),  # sample 10
    np.array([-0.0238, +1.0748, -1.8580, +0.0012, +0.7908, -2.0312, +0.0027, +0.7919, -2.0279, +0.0324, +1.0283, -1.8244]),  # sample 11
    np.array([-0.0261, +1.0708, -1.8621, +0.0011, +0.8215, -2.0559, +0.0033, +0.8221, -2.0527, +0.0363, +1.0176, -1.8225]),  # sample 12
    np.array([-0.0285, +1.0653, -1.8660, +0.0009, +0.8558, -2.0790, +0.0039, +0.8559, -2.0758, +0.0405, +1.0054, -1.8202]),  # sample 13
    np.array([-0.0312, +1.0582, -1.8695, +0.0009, +0.8932, -2.1001, +0.0043, +0.8928, -2.0970, +0.0448, +0.9916, -1.8175]),  # sample 14
    np.array([-0.0339, +1.0492, -1.8728, +0.0009, +0.9330, -2.1190, +0.0046, +0.9321, -2.1159, +0.0492, +0.9764, -1.8146]),  # sample 15
    np.array([-0.0368, +1.0383, -1.8755, +0.0010, +0.9744, -2.1349, +0.0048, +0.9730, -2.1318, +0.0537, +0.9598, -1.8116]),  # sample 16
    np.array([-0.0398, +1.0255, -1.8777, +0.0012, +1.0159, -2.1469, +0.0049, +1.0141, -2.1439, +0.0581, +0.9420, -1.8086]),  # sample 17
    np.array([-0.0427, +1.0108, -1.8793, +0.0016, +1.0560, -2.1540, +0.0048, +1.0538, -2.1511, +0.0622, +0.9230, -1.8058]),  # sample 18
    np.array([-0.0455, +0.9943, -1.8801, +0.0022, +1.0926, -2.1553, +0.0043, +1.0900, -2.1525, +0.0659, +0.9030, -1.8034]),  # sample 19
    np.array([-0.0482, +0.9761, -1.8803, +0.0033, +1.1233, -2.1491, +0.0033, +1.1203, -2.1464, +0.0688, +0.8821, -1.8013]),  # sample 20
    np.array([-0.0503, +0.9565, -1.8798, +0.0047, +1.1456, -2.1340, +0.0019, +1.1424, -2.1314, +0.0709, +0.8608, -1.7996]),  # sample 21
    np.array([-0.0519, +0.9358, -1.8786, +0.0064, +1.1582, -2.1090, +0.0002, +1.1548, -2.1066, +0.0721, +0.8394, -1.7983]),  # sample 22
    np.array([-0.0527, +0.9144, -1.8765, +0.0083, +1.1605, -2.0745, -0.0015, +1.1570, -2.0722, +0.0711, +0.8186, -1.7955]),  # sample 23
    np.array([-0.0532, +0.8931, -1.8731, +0.0098, +1.1556, -2.0490, -0.0040, +1.1506, -2.0305, +0.0704, +0.7998, -1.7951]),  # sample 24
    np.array([-0.0543, +0.8726, -1.8661, +0.0097, +1.1496, -2.0455, -0.0074, +1.1407, -1.9883, +0.0677, +0.7844, -1.7948]),  # sample 25
    np.array([-0.0553, +0.8526, -1.8569, +0.0083, +1.1432, -2.0466, -0.0101, +1.1307, -1.9510, +0.0631, +0.7726, -1.7946]),  # sample 26
    np.array([-0.0533, +0.8327, -1.8522, +0.0063, +1.1371, -2.0464, -0.0123, +1.1213, -1.9189, +0.0559, +0.7630, -1.7979]),  # sample 27
    np.array([-0.0474, +0.8145, -1.8555, +0.0034, +1.1318, -2.0426, -0.0145, +1.1126, -1.8915, +0.0476, +0.7558, -1.8091]),  # sample 28
    np.array([-0.0401, +0.8000, -1.8675, -0.0004, +1.1270, -2.0350, -0.0163, +1.1039, -1.8680, +0.0398, +0.7513, -1.8282]),  # sample 29
    np.array([-0.0332, +0.7898, -1.8861, -0.0048, +1.1229, -2.0241, -0.0176, +1.0946, -1.8480, +0.0334, +0.7496, -1.8529]),  # sample 30
    np.array([-0.0273, +0.7842, -1.9091, -0.0093, +1.1193, -2.0105, -0.0183, +1.0843, -1.8311, +0.0284, +0.7509, -1.8811]),  # sample 31
    np.array([-0.0225, +0.7832, -1.9345, -0.0137, +1.1161, -1.9948, -0.0186, +1.0729, -1.8168, +0.0246, +0.7557, -1.9110]),  # sample 32
    np.array([-0.0188, +0.7870, -1.9609, -0.0176, +1.1131, -1.9778, -0.0187, +1.0602, -1.8048, +0.0217, +0.7646, -1.9413]),  # sample 33
    np.array([-0.0159, +0.7963, -1.9874, -0.0202, +1.1096, -1.9606, -0.0187, +1.0463, -1.7948, +0.0193, +0.7780, -1.9710]),  # sample 34
    np.array([-0.0139, +0.8112, -2.0134, -0.0215, +1.1047, -1.9441, -0.0187, +1.0312, -1.7865, +0.0173, +0.7961, -1.9996]),  # sample 35
    np.array([-0.0126, +0.8314, -2.0383, -0.0214, +1.0975, -1.9290, -0.0187, +1.0151, -1.7796, +0.0154, +0.8189, -2.0267]),  # sample 36
    np.array([-0.0118, +0.8567, -2.0621, -0.0194, +1.0853, -1.9129, -0.0187, +0.9979, -1.7740, +0.0138, +0.8462, -2.0521]),  # sample 37
    np.array([-0.0116, +0.8867, -2.0846, -0.0146, +1.0611, -1.8901, -0.0187, +0.9796, -1.7692, +0.0129, +0.8775, -2.0760]),  # sample 38
    np.array([-0.0101, +0.9191, -2.1052, -0.0076, +1.0323, -1.8692, -0.0248, +0.9734, -1.7815, +0.0125, +0.9115, -2.0979]),  # sample 39
    np.array([-0.0072, +0.9530, -2.1235, +0.0013, +1.0069, -1.8577, -0.0337, +0.9711, -1.8001, +0.0116, +0.9469, -2.1172]),  # sample 40
    np.array([-0.0047, +0.9888, -2.1389, +0.0112, +0.9824, -1.8489, -0.0415, +0.9639, -1.8156, +0.0101, +0.9840, -2.1334]),  # sample 41
    np.array([-0.0034, +1.0262, -2.1506, +0.0218, +0.9643, -1.8489, -0.0482, +0.9507, -1.8263, +0.0078, +1.0219, -2.1454]),  # sample 42
    np.array([-0.0030, +1.0634, -2.1573, +0.0324, +0.9476, -1.8504, -0.0540, +0.9327, -1.8324, +0.0057, +1.0592, -2.1523]),  # sample 43
    np.array([-0.0035, +1.0981, -2.1583, +0.0422, +0.9303, -1.8519, -0.0591, +0.9115, -1.8350, +0.0043, +1.0936, -2.1535]),  # sample 44
    np.array([-0.0048, +1.1276, -2.1518, +0.0507, +0.9121, -1.8531, -0.0638, +0.8883, -1.8352, +0.0036, +1.1226, -2.1472]),  # sample 45
    np.array([-0.0065, +1.1492, -2.1363, +0.0577, +0.8931, -1.8540, -0.0682, +0.8640, -1.8339, +0.0037, +1.1439, -2.1320]),  # sample 46
    np.array([-0.0084, +1.1613, -2.1111, +0.0633, +0.8738, -1.8545, -0.0724, +0.8395, -1.8319, +0.0044, +1.1558, -2.1071]),  # sample 47
    np.array([-0.0102, +1.1631, -2.0762, +0.0675, +0.8544, -1.8549, -0.0762, +0.8153, -1.8299, +0.0049, +1.1568, -2.0793]),  # sample 48
    np.array([-0.0121, +1.1556, -2.0342, +0.0692, +0.8335, -1.8550, -0.0785, +0.7910, -1.8277, +0.0024, +1.1463, -2.0702]),  # sample 49
]




WALK_PHASES: list[np.ndarray] = SAMPLED_JOINT_ANGLES


def get_walk_phases() -> list[np.ndarray]:
    return [phase.copy() for phase in WALK_PHASES]


@dataclass
class GaitParams:
    stride_length: float = 0.10
    swing_height: float = 0.08
    body_height: float = 0.27
    stance_penetration: float = 0.005
    stance_fraction: float = 0.5
    leg_phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)


HOME_FOOT_X = np.array([0.1934, 0.1934, -0.1934, -0.1934])


class BezierGaitPlanner:
    _SWING_CONTROL_POINTS = np.array([
        [-0.50, 0.00],
        [-0.65, 0.00],
        [-0.65, 0.90],
        [-0.65, 0.90],
        [-0.65, 0.90],
        [0.00, 0.90],
        [0.00, 0.90],
        [0.00, 1.20],
        [0.65, 1.20],
        [0.65, 1.20],
        [0.50, 0.00],
        [0.50, 0.00],
    ])

    def __init__(self, params: GaitParams | None = None):
        self.params = params or GaitParams()

    def foot_targets(self, phi: float) -> np.ndarray:
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
        s = 0.5 * (1.0 + np.cos(np.pi * phi_st))
        foot_x = self.params.stride_length * (s - 0.5)
        push = self.params.stance_penetration * np.sin(np.pi * phi_st)
        foot_z = -self.params.body_height - push
        return float(foot_x), float(foot_z)

    def _swing(self, phi_sw: float) -> tuple[float, float]:
        n = len(self._SWING_CONTROL_POINTS) - 1
        point = np.zeros(2)

        for k, pk in enumerate(self._SWING_CONTROL_POINTS):
            coeff = comb(n, k) * (phi_sw ** k) * ((1.0 - phi_sw) ** (n - k))
            point += coeff * pk

        foot_x = self.params.stride_length * point[0]
        foot_z = -self.params.body_height + self.params.swing_height * point[1]
        return float(foot_x), float(foot_z)


def extract_keyframes(
    planner: BezierGaitPlanner,
    n_keyframes: int = 8,
    xml_path: str = "go2/scene.xml",
) -> list[np.ndarray]:
    import mujoco
    from world.kinematics import Go2Kinematics

    model = mujoco.MjModel.from_xml_path(xml_path)
    kin = Go2Kinematics(model)

    keyframes = []
    for i in range(n_keyframes):
        phi = i / n_keyframes
        foot_targets = planner.foot_targets(phi)
        keyframes.append(kin.policy_to_joints(foot_targets))

    return keyframes


def _format_walk_phases_literal(keyframes: list[np.ndarray]) -> str:
    lines = ["WALK_PHASES: list[np.ndarray] = ["]
    for i, kf in enumerate(keyframes):
        nums = ", ".join(f"{x:+.4f}" for x in kf)
        lines.append(f"    np.array([{nums}]),  # phase {i}")
    lines.append("]")
    return "\n".join(lines)


def _format_sampled_joint_angles(samples: list[np.ndarray]) -> str:
    lines = ["SAMPLED_JOINT_ANGLES: list[np.ndarray] = ["]
    for i, q in enumerate(samples):
        nums = ", ".join(f"{x:+.4f}" for x in q)
        lines.append(f"    np.array([{nums}]),  # sample {i}")
    lines.append("]")
    return "\n".join(lines)


def _run_live_planner(
    planner: BezierGaitPlanner,
    cycle_period: float,
    total_duration: float,
    xml_path: str,
    kp: float,
    kd: float,
    sample_count: int | None = None,
) -> None:
    import time
    import mujoco
    import mujoco.viewer
    from env.env import Go2Env
    from world.kinematics import Go2Kinematics

    model_for_kin = mujoco.MjModel.from_xml_path(xml_path)
    kin = Go2Kinematics(model_for_kin)

    initial_targets = planner.foot_targets(0.0)
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

    kin = Go2Kinematics(env.model)
    dt = env.model.opt.timestep * env.control_substeps

    sampled_joint_angles: list[np.ndarray] = []
    sample_times = (
        np.linspace(0.0, total_duration, sample_count, endpoint=False)
        if sample_count is not None
        else None
    )
    next_sample_idx = 0

    print("\nOpening viewer: live planner mode.")
    print(f"  cycle_period={cycle_period}s")
    print(f"  duration={total_duration}s")
    print(f"  dt={dt * 1000:.1f}ms")
    print(f"  sample_count={sample_count}")

    settle_steps = int(cycle_period / dt)
    for k in range(settle_steps):
        phi = (k * dt / cycle_period) % 1.0
        foot_targets = planner.foot_targets(phi)
        joint_targets = kin.policy_to_joints(foot_targets)
        env.step(joint_targets)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        elapsed = 0.0

        while elapsed < total_duration:
            phi = (elapsed / cycle_period) % 1.0
            foot_targets = planner.foot_targets(phi)
            joint_targets = kin.policy_to_joints(foot_targets)

            env.step(joint_targets)

            if sample_times is not None and next_sample_idx < len(sample_times):
                if elapsed >= sample_times[next_sample_idx]:
                    current_joints = env.data.qpos[env._qpos_idx].copy()
                    sampled_joint_angles.append(current_joints)

                    print(
                        f"\nSample {next_sample_idx} "
                        f"at t={elapsed:.3f}s, phi={phi:.3f}:"
                    )
                    print(np.array2string(current_joints, precision=4, separator=", "))

                    next_sample_idx += 1

            if viewer.is_running():
                viewer.sync()

            time.sleep(dt)
            elapsed += dt

        if sampled_joint_angles:
            print("\n" + "=" * 70)
            print("Sampled current joint angles from simulation:")
            print("=" * 70)
            print(_format_sampled_joint_angles(sampled_joint_angles))

        if viewer.is_running():
            final = env.data.qpos[env._qpos_idx].copy()
            while viewer.is_running():
                env.step(final)
                viewer.sync()
                time.sleep(dt)

def _run_manual_sampler(
    planner: BezierGaitPlanner,
    cycle_period: float = 0.4,
    xml_path: str = "go2/scene.xml",
    kp: float = 80.0,
    kd: float = 4.0,
) -> None:
    import time
    import threading
    import queue
    import mujoco
    import mujoco.viewer
    from env.env import Go2Env
    from world.kinematics import Go2Kinematics

    command_queue: queue.Queue[str] = queue.Queue()

    def terminal_input_worker() -> None:
        while True:
            cmd = input()
            command_queue.put(cmd.strip().lower())
            if cmd.strip().lower() == "q":
                break

    input_thread = threading.Thread(target=terminal_input_worker, daemon=True)
    input_thread.start()

    model_for_kin = mujoco.MjModel.from_xml_path(xml_path)
    kin = Go2Kinematics(model_for_kin)

    initial_targets = planner.foot_targets(0.0)
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

    kin = Go2Kinematics(env.model)
    dt = env.model.opt.timestep * env.control_substeps

    saved_samples: list[np.ndarray] = []

    print("\nManual sampling mode.")
    print("Watch the MuJoCo viewer.")
    print("Press ENTER in this terminal to save the current pose.")
    print("Type q then ENTER to quit and print saved samples.")
    print("Joint order:")
    print("FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf")

    settle_steps = int(cycle_period / dt)
    for k in range(settle_steps):
        phi = (k * dt / cycle_period) % 1.0
        foot_targets = planner.foot_targets(phi)
        joint_targets = kin.policy_to_joints(foot_targets)
        env.step(joint_targets)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        elapsed = 0.0
        running = True

        while running and viewer.is_running():
            phi = (elapsed / cycle_period) % 1.0
            foot_targets = planner.foot_targets(phi)
            joint_targets = kin.policy_to_joints(foot_targets)

            env.step(joint_targets)

            while not command_queue.empty():
                cmd = command_queue.get()

                if cmd == "q":
                    running = False
                    break

                current_joints = env.data.qpos[env._qpos_idx].copy()
                saved_samples.append(current_joints)

                print(f"\nSaved sample {len(saved_samples) - 1}")
                print(f"t={elapsed:.3f}s, phi={phi:.3f}")
                print(np.array2string(current_joints, precision=4, separator=", "))

            viewer.sync()
            time.sleep(dt)
            elapsed += dt

    print("\n" + "=" * 70)
    print("Manually selected joint-angle samples:")
    print("=" * 70)

    if saved_samples:
        print(_format_sampled_joint_angles(saved_samples))
    else:
        print("No samples saved.")


def _browse_recorded_cycle(
    planner: BezierGaitPlanner,
    cycle_period: float = 0.4,
    xml_path: str = "go2/scene.xml",
    kp: float = 80.0,
    kd: float = 4.0,
) -> None:
    import time
    import mujoco
    import mujoco.viewer
    from env.env import Go2Env
    from world.kinematics import Go2Kinematics

    model_for_kin = mujoco.MjModel.from_xml_path(xml_path)
    kin = Go2Kinematics(model_for_kin)

    initial_targets = planner.foot_targets(0.0)
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

    kin = Go2Kinematics(env.model)
    dt = env.model.opt.timestep * env.control_substeps
    n_steps = int(cycle_period / dt)

    recorded_qpos: list[np.ndarray] = []
    recorded_qvel: list[np.ndarray] = []
    recorded_joint_angles: list[np.ndarray] = []

    print("\nRecording one full gait cycle...")
    print(f"  cycle_period={cycle_period}")
    print(f"  dt={dt}")
    print(f"  frames={n_steps}")

    for step in range(n_steps):
        elapsed = step * dt
        phi = (elapsed / cycle_period) % 1.0

        foot_targets = planner.foot_targets(phi)
        joint_targets = kin.policy_to_joints(foot_targets)

        env.step(joint_targets)

        recorded_qpos.append(env.data.qpos.copy())
        recorded_qvel.append(env.data.qvel.copy())
        recorded_joint_angles.append(env.data.qpos[env._qpos_idx].copy())

    print("\nRecorded cycle.")
    print("Controls:")
    print("  n       next frame")
    print("  p       previous frame")
    print("  s       save current frame")
    print("  j NUM   jump to frame NUM")
    print("  q       quit and print saved poses")
    print("\nImportant: click the terminal, type command, press ENTER.")

    saved_samples: list[np.ndarray] = []
    frame_idx = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        running = True

        while running and viewer.is_running():
            env.data.qpos[:] = recorded_qpos[frame_idx]
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            viewer.sync()

            print(
                f"\nFrame {frame_idx}/{n_steps - 1} | "
                f"phase={frame_idx / n_steps:.3f}"
            )
            print("Command [n/p/s/j NUM/q]: ", end="", flush=True)

            cmd = input().strip().lower()

            if cmd == "n":
                frame_idx = min(frame_idx + 1, n_steps - 1)

            elif cmd == "p":
                frame_idx = max(frame_idx - 1, 0)

            elif cmd == "s":
                sample = recorded_joint_angles[frame_idx].copy()
                saved_samples.append(sample)

                print(f"Saved frame {frame_idx} as sample {len(saved_samples) - 1}")
                print(np.array2string(sample, precision=4, separator=", "))

            elif cmd.startswith("j "):
                try:
                    target = int(cmd.split()[1])
                    frame_idx = max(0, min(target, n_steps - 1))
                except ValueError:
                    print("Invalid jump command. Use: j 120")

            elif cmd == "q":
                running = False

            else:
                print("Unknown command.")

            time.sleep(0.02)

    print("\n" + "=" * 70)
    print("Selected clean poses:")
    print("=" * 70)

    if saved_samples:
        print(_format_sampled_joint_angles(saved_samples))
    else:
        print("No samples saved.")



def _run_keyframes_in_viewer(
    keyframes: list[np.ndarray],
    cycle_period: float,
    total_duration: float,
    xml_path: str,
    kp: float,
    kd: float,
    backward: bool = False,
) -> None:
    import time
    import mujoco
    import mujoco.viewer
    from env.env import Go2Env
    from world.trajectory import make_trajectory, trajectory_duration_to_nsteps

    if backward:
        keyframes = list(reversed(keyframes))

    env = Go2Env(
        xml_path=xml_path,
        control_substeps=4,
        kp=kp,
        kd=kd,
        initial_base_height=0.27,
        initial_angles=list(keyframes[0]),
        settle_steps=500,
    )
    env.reset()

    n_phases = len(keyframes)
    phase_duration = cycle_period / n_phases
    dt = env.model.opt.timestep * env.control_substeps
    n_steps_per_phase = trajectory_duration_to_nsteps(phase_duration, dt)

    direction = "backward" if backward else "forward"

    print(f"\nOpening viewer: keyframe replay mode, {direction}.")
    print(f"  keyframes={n_phases}")
    print(f"  cycle_period={cycle_period}s")
    print(f"  phase_duration={phase_duration:.4f}s")
    print(f"  duration={total_duration}s")

    for phase_idx in range(n_phases):
        start_joints = env.data.qpos[env._qpos_idx].copy()
        target_joints = keyframes[phase_idx]
        traj = make_trajectory(start_joints, target_joints, phase_duration)

        for step in range(n_steps_per_phase):
            t = step * dt
            env.step(traj(t))

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        elapsed = 0.0
        phase_idx = 0

        while viewer.is_running() and elapsed < total_duration:
            start_joints = env.data.qpos[env._qpos_idx].copy()
            target_joints = keyframes[phase_idx]
            traj = make_trajectory(start_joints, target_joints, phase_duration)

            for step in range(n_steps_per_phase):
                if not viewer.is_running() or elapsed >= total_duration:
                    break

                t = step * dt
                env.step(traj(t))
                viewer.sync()
                time.sleep(dt)
                elapsed += dt

            phase_idx = (phase_idx + 1) % n_phases

        final = env.data.qpos[env._qpos_idx].copy()
        while viewer.is_running():
            env.step(final)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize Go2 walking gait.",
    )

    parser.add_argument("--cycle-period", type=float, default=0.4)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--stride", type=float, default=None)
    parser.add_argument("--swing-height", type=float, default=None)
    parser.add_argument("--kp", type=float, default=80.0)
    parser.add_argument("--kd", type=float, default=4.0)
    parser.add_argument("--xml", type=str, default="go2/scene.xml")

    parser.add_argument(
        "--extract",
        type=int,
        default=None,
        metavar="N",
        help="Extract N IK keyframes from the Bezier planner and replay them.",
    )

    parser.add_argument(
        "--browse-cycle",
        action="store_true",
        help="Record one gait cycle, then step through each pose manually and save clean ones.",
    )

    parser.add_argument(
        "--sample-current",
        type=int,
        default=None,
        metavar="N",
        help="Run live planner and sample actual simulated joint angles N times.",
    )

    parser.add_argument(
        "--replay-sampled",
        action="store_true",
        help="Replay the hardcoded SAMPLED_JOINT_ANGLES / WALK_PHASES.",
    )

    parser.add_argument(
        "--backward",
        action="store_true",
        help="Replay keyframes backward.",
    )

    parser.add_argument(
        "--manual-sample",
        action="store_true",
        help="Run the gait and manually save current joint angles by pressing ENTER.",
    )   

    args = parser.parse_args()

    params = GaitParams()

    if args.stride is not None:
        params.stride_length = args.stride

    if args.swing_height is not None:
        params.swing_height = args.swing_height

    planner = BezierGaitPlanner(params)

    if args.browse_cycle:
        _browse_recorded_cycle(
            planner,
            cycle_period=args.cycle_period,
            xml_path=args.xml,
            kp=args.kp,
            kd=args.kd,
        )

    if args.manual_sample:
        _run_manual_sampler(
            planner,
            cycle_period=args.cycle_period,
            xml_path=args.xml,
            kp=args.kp,
            kd=args.kd,
        )

    if args.replay_sampled:
        _run_keyframes_in_viewer(
            WALK_PHASES,
            cycle_period=args.cycle_period,
            total_duration=args.duration,
            xml_path=args.xml,
            kp=args.kp,
            kd=args.kd,
            backward=args.backward,
        )

    elif args.extract is not None:
        keyframes = extract_keyframes(
            planner,
            n_keyframes=args.extract,
            xml_path=args.xml,
        )

        print("\n" + "=" * 70)
        print("Paste this over WALK_PHASES:")
        print("=" * 70)
        print(_format_walk_phases_literal(keyframes))

        _run_keyframes_in_viewer(
            keyframes,
            cycle_period=args.cycle_period,
            total_duration=args.duration,
            xml_path=args.xml,
            kp=args.kp,
            kd=args.kd,
            backward=args.backward,
        )

    else:
        if args.backward:
            print("Warning: --backward only affects keyframe replay mode.")

        _run_live_planner(
            planner,
            cycle_period=args.cycle_period,
            total_duration=args.duration,
            xml_path=args.xml,
            kp=args.kp,
            kd=args.kd,
            sample_count=args.sample_current,
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