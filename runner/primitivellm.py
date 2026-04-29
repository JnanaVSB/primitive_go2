"""FORGE-style policy search loop for code-as-policy.

Supports two modes:
  1. Single task: config has target + distance_weight. Reward from final state.
  2. Checkpoint sequence: config has checkpoints list. Per-task rewards from
     robot.checkpoint() calls made by the LLM's code.

Artifacts per run:
    logs/<task>_<ISO-timestamp>/
        config.json       — exact config used
        trial_log.json    — trial history, updated after each iteration
        run.log           — structured log (LLM responses, rewards, errors)
        iter_XX.mp4       — video of each iteration
"""

import json
import re
import time
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import Config
from env.env import Go2Env
from env.reward import compute_pose_reward
from world.robot_api import RobotAPI
from world.kinematics import Go2Kinematics
from runner.code_executor import execute_policy_code
from agent.prompt import PromptBuilder
from agent.llm_agents import make_client
from runner.trial_log import TrialLog
from runner.recorder import RenderingEnv

logger = logging.getLogger(__name__)

_PARSE_FAIL_REWARD = -10.0
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)


def run(cfg: Config, resume_log_path: str | None = None) -> TrialLog:
    """Execute the FORGE loop."""
    run_dir = _setup_run_dir(cfg, resume_log_path)
    _attach_run_log_file(run_dir)
    trial_log = _load_or_create_log(resume_log_path)

    success_threshold = cfg.runner.success_threshold

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Task: {cfg.task.name}")
    if cfg.task.has_checkpoints:
        cp_names = [cp.name for cp in cfg.task.checkpoints]
        logger.info(f"Checkpoints: {cp_names}")
        print(f"Task: {cfg.task.name} | Checkpoints: {cp_names}")
    else:
        print(f"Task: {cfg.task.name}")
    logger.info(f"Starting from iteration {len(trial_log) + 1}/{cfg.runner.max_iterations}")
    print(f"Iterations: {cfg.runner.max_iterations}")

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
        print(f"\n[iter {i}/{cfg.runner.max_iterations}] prompting LLM...")

        # Build prompt
        prompt = prompt_builder.build(
            task=cfg.task.name,
            iter_idx=i,
            max_iters=cfg.runner.max_iterations,
            trial_history=trial_log.to_prompt_records(),
        )

        # Get code from LLM with retries
        code, rationale = _generate_with_parse_retry(
            llm, prompt, cfg.runner.max_parse_retries, i,
        )
        if code is None:
            trial_log.append(
                code="# parse failure",
                reward=_PARSE_FAIL_REWARD,
                rationale=rationale,
            )
            trial_log.save(run_dir / "trial_log.json")
            print(f"[iter {i}] parse failure, reward={_PARSE_FAIL_REWARD}")
            continue

        logger.info(f"[iter {i}] LLM rationale: {rationale}")
        logger.info(f"[iter {i}] LLM code:\n{code}")

        # Execute code and compute reward(s)
        reward, per_task_rewards, exec_error = _execute_and_reward(
            cfg, code, run_dir / f"iter_{i:02d}.mp4",
        )

        if exec_error:
            logger.warning(f"[iter {i}] execution error: {exec_error}")
            rationale += f"\n\nExecution error: {exec_error}"

        # Format reward string for display and trial log
        if per_task_rewards:
            reward_str = ", ".join(
                f"{name}: {r:.4f}" for name, r in per_task_rewards.items()
            )
            reward_str += f" | total: {reward:.4f}"
            print(f"[iter {i}] {reward_str}")
            logger.info(f"[iter {i}] {reward_str}")
        else:
            print(f"[iter {i}] reward={reward:.4f}")
            logger.info(f"[iter {i}] reward={reward:.4f}")

        trial_log.append(
            code=code,
            reward=reward,
            rationale=rationale,
            per_task_rewards=per_task_rewards,
        )
        trial_log.save(run_dir / "trial_log.json")

        passed = reward >= success_threshold
        if passed:
            print(f"\nTask passed at iteration {i}!")
            logger.info(f"Task passed at iteration {i}")
            break

    best = trial_log.best
    if best:
        print(f"\nBest reward: {best.reward:.4f} at iteration {best.iteration}")
        logger.info(f"Best reward: {best.reward:.4f} at iteration {best.iteration}")
    return trial_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_code(response: str) -> tuple[str, str]:
    """Extract Python code block and rationale from LLM response."""
    matches = _CODE_BLOCK_RE.findall(response)
    if not matches:
        raise ValueError("No fenced Python code block found in response.")
    code = "\n".join(m.strip() for m in matches)
    rationale = _CODE_BLOCK_RE.sub("", response).strip()
    return code, rationale


def _generate_with_parse_retry(
    llm, initial_prompt: str, max_retries: int, iteration: int,
) -> tuple[str | None, str]:
    """Call the LLM and extract code; retry on parse failure."""
    prompt = initial_prompt
    for attempt in range(1, max_retries + 1):
        print(f"[iter {iteration}] LLM call (attempt {attempt}/{max_retries})...")
        try:
            response = llm.generate(prompt)
        except Exception as e:
            logger.error(f"[iter {iteration}] LLM call failed: {e}")
            return None, f"LLM error: {e}"

        logger.info(f"[iter {iteration}] LLM response (attempt {attempt}):\n{response}")

        try:
            code, rationale = _extract_code(response)
            return code, rationale
        except ValueError as e:
            logger.warning(f"[iter {iteration}] parse failed (attempt {attempt}): {e}")
            print(f"[iter {iteration}] parse failed: {e}")
            if attempt == max_retries:
                return None, (
                    f"Parse failed after {max_retries} attempts. "
                    f"Last error: {e}\n\nLast response:\n{response[:500]}"
                )
            prompt = (
                initial_prompt
                + f"\n\n# Retry needed\n"
                + f"Your previous response did not parse. Error: {e}\n"
                + f"Make sure your response contains a fenced Python code block "
                + f"(```python ... ```). Try again."
            )
    return None, "unreachable"


def _execute_and_reward(
    cfg: Config,
    code: str,
    video_path: Path,
) -> tuple[float, dict | None, str]:
    """Create env, run code via executor, compute reward(s).

    Returns:
        (total_reward, per_task_rewards_dict_or_None, error_message).
    """
    gains = cfg.stiffness_modes.get('normal', list(cfg.stiffness_modes.values())[0])
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
    env = RenderingEnv(base_env)
    robot = RobotAPI(env)
    kin = Go2Kinematics(base_env.model)

    try:
        result = execute_policy_code(code, robot, kinematics=kin)

        if cfg.task.has_checkpoints:
            total_reward, per_task_rewards = _compute_checkpoint_rewards(
                robot, cfg.task.checkpoints,
            )
        else:
            target = asdict(cfg.task.target) if cfg.task.target else {}
            distance_weight = getattr(cfg.task, 'distance_weight', 0.0)
            state = robot.get_state()
            total_reward = compute_pose_reward(state, target, distance_weight=distance_weight)
            per_task_rewards = None

            logger.info(
                f"Post-execution state: "
                f"x={state['x']:.4f} h={state['h']:.4f} "
                f"roll={state['roll']:.4f} pitch={state['pitch']:.4f} "
                f"reward={total_reward:.4f}"
            )

        env.save_video(video_path)

        if not result.success:
            return total_reward, per_task_rewards, result.error
        return total_reward, per_task_rewards, ""

    finally:
        env.close()


def _compute_checkpoint_rewards(robot, checkpoint_configs) -> tuple[float, dict]:
    """Compute per-task rewards from robot checkpoints.

    Matches checkpoint names from the robot to checkpoint configs.
    Distance is computed relative to the previous checkpoint's x position.

    Returns:
        (total_reward, {name: reward} dict).
    """
    per_task = {}
    prev_x = 0.0  # starting x position

    for cp_cfg in checkpoint_configs:
        # Find matching checkpoint from robot
        matching = [cp for cp in robot.checkpoints if cp['name'] == cp_cfg.name]

        if not matching:
            # Checkpoint not called — penalty
            per_task[cp_cfg.name] = _PARSE_FAIL_REWARD
            logger.warning(f"Checkpoint '{cp_cfg.name}' not found in robot checkpoints")
            continue

        # Use the last one if called multiple times
        cp_state = matching[-1]['state']

        # Compute distance since last checkpoint
        distance_since_last = cp_state['x'] - prev_x

        # Build target dict
        target = asdict(cp_cfg.target)

        # Compute reward — use distance_since_last instead of absolute x
        state_for_reward = dict(cp_state)
        state_for_reward['x'] = distance_since_last

        reward = compute_pose_reward(
            state_for_reward, target, distance_weight=cp_cfg.distance_weight,
        )

        per_task[cp_cfg.name] = reward
        prev_x = cp_state['x']

        logger.info(
            f"Checkpoint '{cp_cfg.name}': "
            f"x={cp_state['x']:.4f} (dist={distance_since_last:.4f}) "
            f"h={cp_state['h']:.4f} roll={cp_state['roll']:.4f} "
            f"pitch={cp_state['pitch']:.4f} reward={reward:.4f}"
        )

    total = sum(per_task.values()) / len(per_task) if per_task else _PARSE_FAIL_REWARD
    return total, per_task


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
        'task': {
            'name': cfg.task.name,
        },
    }
    if cfg.task.target:
        d['task']['target'] = asdict(cfg.task.target)
    if cfg.task.has_checkpoints:
        d['task']['checkpoints'] = [
            {'name': cp.name, 'target': asdict(cp.target), 'distance_weight': cp.distance_weight}
            for cp in cfg.task.checkpoints
        ]
    return d