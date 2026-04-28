"""FORGE-style policy search loop for primitive_go2.

Iteratively prompts an LLM to produce Policy/Policies, executes in MuJoCo,
scores by pose distance, and feeds the trial history back as context.

Supports both single-task runs (sit, lay) and sequence runs (lay→stand, sit→stand).

Artifacts per run:
    logs/<task>_<ISO-timestamp>/
        config.json       — exact config used
        trial_log.json    — trial history, updated after each iteration
        run.log           — structured log (LLM responses, rewards, errors)
        iter_XX.mp4       — video of each iteration
"""

import json
import time
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import Config
from env.env import Go2Env
from env.reward import compute_pose_reward
from world.kinematics import Go2Kinematics
from world.primitive import execute_policy, extract_base_pose
from world.trajectory import make_trajectory, trajectory_duration_to_nsteps
from agent.policy import Policy
from agent.prompt import PromptBuilder
from agent.parser import parse_response, ParseError
from agent.llm_agents import make_client
from runner.keyframe_trial_log import TrialLog
from runner.recorder import RenderingEnv

logger = logging.getLogger(__name__)

# Fallback policy on parse failure — a no-op standing pose.
_FALLBACK_POLICY = Policy(
    foot_targets=np.array([
        [ 0.1934, -0.27], [ 0.1934, -0.27],
        [-0.1934, -0.27], [-0.1934, -0.27],
    ]),
    duration=1.0,
    stiffness='normal',
)
_PARSE_FAIL_REWARD = -10.0


def run(cfg: Config, resume_log_path: str | None = None) -> TrialLog:
    """Execute the FORGE loop."""
    run_dir = _setup_run_dir(cfg, resume_log_path)
    _attach_run_log_file(run_dir)
    trial_log = _load_or_create_log(resume_log_path)

    steps = cfg.task.steps
    num_steps = len(steps)
    step_names = [s.name for s in steps]
    total_policy_count = sum(s.policy_count for s in steps)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Task: {cfg.task.name}, steps: {step_names}")
    logger.info(f"Starting from iteration {len(trial_log) + 1}/{cfg.runner.max_iterations}")
    print(f"Task: {cfg.task.name} | Steps: {step_names} | Iterations: {cfg.runner.max_iterations}")

    prompt_builder = PromptBuilder(cfg.runner.templates_dir)
    llm = make_client(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        max_retries=cfg.llm.max_retries,
        retry_delay=cfg.llm.retry_delay,
        **({'base_url': cfg.llm.base_url} if cfg.llm.base_url else {}),
    )

    start_i = len(trial_log) + 1
    for i in range(start_i, cfg.runner.max_iterations + 1):
        print(f"[iter {i}/{cfg.runner.max_iterations}] prompting LLM...")

        prompt = prompt_builder.build(
            task=cfg.task.name,
            iter_idx=i,
            max_iters=cfg.runner.max_iterations,
            trial_history=trial_log.to_prompt_records(),
        )

        policies, rationale = _generate_with_parse_retry(
            llm, prompt, cfg.runner.max_parse_retries,
            i, cfg.llm.provider, cfg.llm.model,
            expected_count=total_policy_count,
        )
        if policies is None:
            fallback_policies = [_FALLBACK_POLICY] * total_policy_count
            fallback_rewards = [_PARSE_FAIL_REWARD] * num_steps
            trial_log.append(fallback_policies, fallback_rewards, rationale, step_names)
            trial_log.save(run_dir / "trial_log.json")
            continue

        logger.info(f"[iter {i}] LLM rationale: {rationale}")
        for idx, pol in enumerate(policies):
            name = step_names[idx] if idx < len(step_names) else f"step_{idx+1}"
            _log_policy(i, pol, name)

        rewards = _execute_sequence_and_record(
            cfg, policies, steps, run_dir / f"iter_{i:02d}.mp4",
        )

        trial_log.append(policies, rewards, rationale, step_names)
        trial_log.save(run_dir / "trial_log.json")

        # Print rewards
        reward_parts = []
        for idx, (name, rew) in enumerate(zip(step_names, rewards)):
            passed = rew >= steps[idx].success_threshold
            status = "PASS" if passed else ""
            reward_parts.append(f"{name}={rew:.4f}{' '+status if passed else ''}")
        print(f"[iter {i}/{cfg.runner.max_iterations}] {' | '.join(reward_parts)}")

        logger.info(f"[iter {i}] rewards: {list(zip(step_names, rewards))}")

        # Check if all steps passed their thresholds
        all_passed = all(
            rew >= step.success_threshold
            for rew, step in zip(rewards, steps)
        )
        if all_passed:
            print(f"\nAll steps passed at iteration {i}!")
            logger.info(f"All steps passed at iteration {i}")
            break

    best = trial_log.best
    if best:
        print(f"\nBest total reward: {best.reward:.4f} at iteration {best.iteration}")
        logger.info(f"Best total reward: {best.reward:.4f} at iteration {best.iteration}")
    return trial_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_with_parse_retry(
    llm, initial_prompt: str, max_parse_retries: int,
    iteration: int, provider: str, model: str,
    expected_count: int = 1,
) -> tuple[list[Policy] | None, str]:
    """Call the LLM and parse; retry on parse failure up to N times."""
    prompt = initial_prompt
    for attempt in range(1, max_parse_retries + 1):
        print(f"[iter {iteration}] LLM call (attempt {attempt}/{max_parse_retries})...")
        try:
            response = llm.generate(prompt)
        except Exception as e:
            logger.error(f"[iter {iteration}] LLM call failed: {e}")
            return None, f"LLM error: {e}"

        logger.info(f"[iter {iteration}] LLM response (attempt {attempt}):\n{response}")

        try:
            policies, rationale = parse_response(response, expected_count=expected_count)
            return policies, rationale
        except ParseError as e:
            logger.warning(f"[iter {iteration}] parse failed (attempt {attempt}): {e}")
            print(f"[iter {iteration}] parse failed: {e}")
            if attempt == max_parse_retries:
                return None, (
                    f"parse failed after {max_parse_retries} attempts. "
                    f"Last error: {e}\n\nLast response:\n{response[:500]}"
                )
            policy_word = "Policy" if expected_count == 1 else f"list of {expected_count} Policy objects"
            prompt = (
                initial_prompt
                + f"\n\n# Retry needed\n"
                + f"Your previous response did not parse. Error: {e}\n"
                + f"Make sure your response contains a fenced Python code block "
                + f"with a valid {policy_word}. Try again."
            )
    return None, "unreachable"


def _execute_sequence_and_record(
    cfg: Config,
    policies: list[Policy],
    steps: list,
    video_path: Path,
) -> list[float]:
    """Execute a sequence of policies without resetting between them.

    Each step consumes step.policy_count policies from the list.
    If step.loop_duration > 0, those policies are looped as a gait cycle
    for that many seconds (no settling between cycle repetitions).
    Otherwise the policies are executed once with settling after.

    Returns a list of rewards, one per step.
    """
    # Use the first policy's stiffness for initial env setup
    gains = cfg.stiffness_modes[policies[0].stiffness]
    base_env = Go2Env(
        xml_path=cfg.env.xml_path,
        control_substeps=cfg.env.control_substeps,
        kp=gains.kp,
        kd=gains.kd,
        initial_base_height=cfg.env.initial_base_height,
        initial_angles=cfg.env.initial_angles,
        settle_steps=cfg.env.settle_steps,
    )
    base_env.reset()
    kin = Go2Kinematics(base_env.model)
    env = RenderingEnv(base_env)

    rewards = []
    policy_idx = 0  # tracks which policies we've consumed

    try:
        for step in steps:
            # Slice out this step's policies
            step_policies = policies[policy_idx:policy_idx + step.policy_count]
            policy_idx += step.policy_count

            # Override duration from config
            for pol in step_policies:
                pol.duration = step.phase_duration

            if step.loop_duration > 0:
                # Gait mode: loop the cycle for loop_duration seconds
                base_state = _execute_gait_loop(
                    env, kin, step_policies, step.loop_duration,
                    cfg.primitive.settle_steps_after,
                )
            else:
                # Normal mode: execute policies once with settling
                for pol in step_policies:
                    base_state = execute_policy(
                        env, kin, pol, cfg.primitive.settle_steps_after,
                    )

            reward = compute_pose_reward(
                base_state, asdict(step.target),
                distance_weight=step.distance_weight,
            )
            rewards.append(reward)

            logger.info(
                f"Step ({step.name}): "
                f"x={base_state['x']:.4f} h={base_state['h']:.4f} "
                f"roll={base_state['roll']:.4f} pitch={base_state['pitch']:.4f} "
                f"reward={reward:.4f}"
            )

        env.save_video(video_path)
    finally:
        env.close()

    return rewards


def _execute_gait_loop(
    env,
    kin: Go2Kinematics,
    cycle_policies: list[Policy],
    loop_duration: float,
    settle_steps_after: int,
) -> dict:
    """Execute a gait cycle repeatedly for loop_duration seconds.

    Each policy in the cycle is executed as a trajectory without settling
    between them. The cycle repeats until the total time exceeds loop_duration.
    After the loop ends, the robot settles before measuring pose.

    Returns the base state dict after settling.
    """
    dt_per_control_step = env.model.opt.timestep * env.control_substeps
    elapsed = 0.0

    while elapsed < loop_duration:
        for policy in cycle_policies:
            if elapsed >= loop_duration:
                break

            # IK and trajectory for this gait phase
            start_joints = env.data.qpos[env._qpos_idx].copy()
            target_joints = kin.policy_to_joints(policy.foot_targets)
            traj = make_trajectory(start_joints, target_joints, policy.duration)
            n_steps = trajectory_duration_to_nsteps(policy.duration, dt_per_control_step)

            for step in range(n_steps):
                if elapsed >= loop_duration:
                    break
                t = step * dt_per_control_step
                action = traj(t)
                env.step(action)
                elapsed += dt_per_control_step

    # Settle at the end before measuring
    final_joints = env.data.qpos[env._qpos_idx].copy()
    for _ in range(settle_steps_after):
        env.step(final_joints)

    return extract_base_pose(env, kin)


def _log_policy(iteration: int, policy: Policy, step_name: str):
    """Log policy parameters to file only."""
    ft = policy.foot_targets
    leg_names = ['FR', 'FL', 'RR', 'RL']

    lines = [f"[iter {iteration}] {step_name} policy:"]
    for name, row in zip(leg_names, ft):
        lines.append(f"    {name}: foot_x={row[0]:+.4f}, foot_z={row[1]:+.4f}")

    logger.info("\n".join(lines))


def _setup_run_dir(cfg: Config, resume_log_path: str | None) -> Path:
    if resume_log_path:
        run_dir = Path(resume_log_path).parent
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
    else:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(cfg.runner.log_dir) / f"{cfg.task.name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(_config_to_dict(cfg), f, indent=2)

    return run_dir


def _attach_run_log_file(run_dir: Path):
    """Add a FileHandler for structured logging to run_dir/run.log."""
    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path, mode='a')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    ))
    logging.getLogger().addHandler(handler)
    logger.info(f"--- Run log started. Output dir: {run_dir} ---")


def _load_or_create_log(resume_log_path: str | None) -> TrialLog:
    if not resume_log_path:
        return TrialLog()
    path = Path(resume_log_path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume log not found: {path}")
    log = TrialLog.load(path)
    logger.info(f"Resuming from {path} with {len(log)} existing entries")
    print(f"Resuming with {len(log)} existing entries")
    return log


def _config_to_dict(cfg: Config) -> dict:
    d = {
        'env': asdict(cfg.env),
        'primitive': asdict(cfg.primitive),
        'stiffness_modes': {
            name: asdict(gains) for name, gains in cfg.stiffness_modes.items()
        },
        'llm': asdict(cfg.llm),
        'runner': asdict(cfg.runner),
        'task': {'name': cfg.task.name},
    }
    if cfg.task.is_sequence:
        d['task']['sequence'] = [
            {'name': s.name, 'target': asdict(s.target), 'success_threshold': s.success_threshold}
            for s in cfg.task.sequence
        ]
    else:
        d['task']['target'] = asdict(cfg.task.target)
    return d