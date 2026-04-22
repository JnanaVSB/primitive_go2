"""FORGE-style policy search loop for primitive_go2.

Iteratively prompts an LLM to produce a Policy, executes it in MuJoCo,
scores by pose distance, and feeds the trial history back as context.

Artifacts per run:
    logs/<task>_<ISO-timestamp>/
        config.json       — exact config used
        trial_log.json    — trial history, updated after each iteration
        run.log           — text log of the whole run
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
from world.primitive import execute_policy
from agent.policy import Policy
from agent.prompt import PromptBuilder
from agent.parser import parse_response, ParseError
from agent.llm_agents import make_client
from runner.trial_log import TrialLog
from runner.recorder import RenderingEnv

logger = logging.getLogger(__name__)

# Fallback policy on parse failure — a no-op with a large penalty reward.
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
    """Execute the FORGE loop for one task."""
    run_dir = _setup_run_dir(cfg, resume_log_path)
    _attach_run_log_file(run_dir)
    trial_log = _load_or_create_log(resume_log_path)

    msg_header = (
        f"Run directory: {run_dir}\n"
        f"Task: {cfg.task.name}, target: {asdict(cfg.task.target)}\n"
        f"Starting from iteration {len(trial_log) + 1}/{cfg.runner.max_iterations}"
    )
    print(msg_header)
    logger.info(msg_header)

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
        _log_iter(i, cfg.runner.max_iterations, "building prompt...")
        prompt = prompt_builder.build(
            task=cfg.task.name,
            iter_idx=i,
            max_iters=cfg.runner.max_iterations,
            trial_history=trial_log.to_prompt_records(),
        )

        policy, rationale = _generate_with_parse_retry(
            llm, prompt, cfg.runner.max_parse_retries, i, cfg.llm.provider, cfg.llm.model,
        )
        if policy is None:
            trial_log.append(_FALLBACK_POLICY, _PARSE_FAIL_REWARD, rationale)
            trial_log.save(run_dir / "trial_log.json")
            continue

        _log_policy(i, policy, rationale)

        reward = _execute_and_record(cfg, policy, run_dir / f"iter_{i:02d}.mp4")

        trial_log.append(policy, reward, rationale)
        trial_log.save(run_dir / "trial_log.json")
        _log_iter(i, cfg.runner.max_iterations, f"reward = {reward:.4f}")

        if reward >= cfg.runner.success_threshold:
            msg = (f"Success at iteration {i} "
                   f"(reward {reward:.4f} >= threshold {cfg.runner.success_threshold}).")
            print(f"\n{msg}")
            logger.info(msg)
            break

    best = trial_log.best
    if best:
        msg = f"Best reward: {best.reward:.4f} at iteration {best.iteration}"
        print(f"\n{msg}")
        logger.info(msg)
    return trial_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_with_parse_retry(
    llm, initial_prompt: str, max_parse_retries: int,
    iteration: int, provider: str, model: str,
) -> tuple[Policy | None, str]:
    """Call the LLM and parse; retry on parse failure up to N times."""
    prompt = initial_prompt
    for attempt in range(1, max_parse_retries + 1):
        _log_iter(
            iteration, None,
            f"calling LLM ({provider}/{model}, parse attempt {attempt}/{max_parse_retries})...",
        )
        try:
            response = llm.generate(prompt)
        except Exception as e:
            msg = f"LLM call failed: {e}"
            print(f"[iter {iteration}] {msg}")
            logger.error(f"[iter {iteration}] {msg}")
            return None, f"LLM error: {e}"

        try:
            policy, rationale = parse_response(response)
            return policy, rationale
        except ParseError as e:
            msg = f"parse failed on attempt {attempt}: {e}"
            print(f"[iter {iteration}] {msg}")
            logger.warning(f"[iter {iteration}] {msg}")
            if attempt == max_parse_retries:
                return None, (
                    f"parse failed after {max_parse_retries} attempts. "
                    f"Last error: {e}\n\nLast response:\n{response[:500]}"
                )
            prompt = (
                initial_prompt
                + f"\n\n# Retry needed\n"
                + f"Your previous response did not parse. Error: {e}\n"
                + "Make sure your response contains a fenced Python code block "
                + "with a valid Policy(...) expression. Try again."
            )
    return None, "unreachable"


def _execute_and_record(cfg: Config, policy: Policy, video_path: Path) -> float:
    """Run one policy, record the video, return the reward."""
    gains = cfg.stiffness_modes[policy.stiffness]
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

    try:
        base_state = execute_policy(
            env, kin, policy, cfg.primitive.settle_steps_after,
        )
        reward = compute_pose_reward(base_state, asdict(cfg.task.target))
        env.save_video(video_path)
    finally:
        env.close()

    return reward


def _log_iter(iteration: int, max_iters: int | None, message: str):
    """Print and log an iteration-scoped status message."""
    if max_iters is not None:
        prefix = f"[iter {iteration}/{max_iters}]"
    else:
        prefix = f"[iter {iteration}]"
    line = f"{prefix} {message}"
    print(line)
    logger.info(line)


def _log_policy(iteration: int, policy: Policy, rationale: str):
    """Print/log only policy parameters."""
    ft = policy.foot_targets
    leg_names = ['FR', 'FL', 'RR', 'RL']

    lines = [f"[iter {iteration}] LLM proposed policy:"]

    for name, row in zip(leg_names, ft):
        lines.append(
            f"    {name}: foot_x={row[0]:+.4f}, foot_z={row[1]:+.4f}"
        )

    lines.append(
        f"    duration={policy.duration}, stiffness={policy.stiffness}"
    )

    for line in lines:
        print(line)

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
    """Add a FileHandler so root logger also writes to run_dir/run.log."""
    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path, mode='a')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    ))
    # Attach to the root logger so every module's logger writes here
    logging.getLogger().addHandler(handler)
    logging.getLogger().info(f"--- Run log started. Output dir: {run_dir} ---")


def _load_or_create_log(resume_log_path: str | None) -> TrialLog:
    if not resume_log_path:
        return TrialLog()
    path = Path(resume_log_path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume log not found: {path}")
    log = TrialLog.load(path)
    msg = f"Resuming from {path} with {len(log)} existing entries"
    print(msg)
    logger.info(msg)
    return log


def _config_to_dict(cfg: Config) -> dict:
    return {
        'env': asdict(cfg.env),
        'primitive': asdict(cfg.primitive),
        'stiffness_modes': {
            name: asdict(gains) for name, gains in cfg.stiffness_modes.items()
        },
        'llm': asdict(cfg.llm),
        'runner': asdict(cfg.runner),
        'task': {
            'name': cfg.task.name,
            'target': asdict(cfg.task.target),
        },
    }