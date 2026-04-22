"""FORGE-style policy search loop for primitive_go2.

Iteratively prompts an LLM to produce a Policy, executes it in MuJoCo,
scores by pose distance, and feeds the trial history back as context.

Artifacts per run:
    logs/<task>_<timestamp>/
        config.json       — exact config used
        trial_log.json    — trial history, updated after each iteration
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

# Fallback policy when parsing fails — a no-op (targets home foot positions).
# Paired with a punishing reward so the LLM sees the failure in history.
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
    """Execute the FORGE loop for one task.

    Args:
        cfg: loaded Config.
        resume_log_path: optional path to an existing trial_log.json to continue from.

    Returns:
        The final TrialLog (also saved to disk).
    """
    run_dir = _setup_run_dir(cfg, resume_log_path)
    trial_log = _load_or_create_log(resume_log_path)

    print(f"Run directory: {run_dir}")
    print(f"Task: {cfg.task.name}, target: {asdict(cfg.task.target)}")
    print(f"Starting from iteration {len(trial_log) + 1}/{cfg.runner.max_iterations}")

    prompt_builder = PromptBuilder(cfg.runner.templates_dir)
    llm = make_client(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        **({'base_url': cfg.llm.base_url} if cfg.llm.base_url else {}),
    )

    start_i = len(trial_log) + 1
    for i in range(start_i, cfg.runner.max_iterations + 1):
        print(f"\n[iter {i}/{cfg.runner.max_iterations}] building prompt...")
        prompt = prompt_builder.build(
            task=cfg.task.name,
            iter_idx=i,
            max_iters=cfg.runner.max_iterations,
            trial_history=trial_log.to_prompt_records(),
        )

        print(f"[iter {i}] calling LLM ({cfg.llm.provider}/{cfg.llm.model})...")
        try:
            response = llm.generate(prompt)
        except Exception as e:
            print(f"[iter {i}] LLM call failed: {e}")
            trial_log.append(_FALLBACK_POLICY, _PARSE_FAIL_REWARD, f"LLM error: {e}")
            trial_log.save(run_dir / "trial_log.json")
            continue

        try:
            policy, rationale = parse_response(response)
        except ParseError as e:
            print(f"[iter {i}] parse failed: {e}")
            trial_log.append(
                _FALLBACK_POLICY, _PARSE_FAIL_REWARD,
                f"parse failed: {e}\n\nRaw response:\n{response[:500]}",
            )
            trial_log.save(run_dir / "trial_log.json")
            continue

        print(f"[iter {i}] executing policy (stiffness={policy.stiffness}, dur={policy.duration})...")
        reward = _execute_and_record(cfg, policy, run_dir / f"iter_{i:02d}.mp4")

        trial_log.append(policy, reward, rationale)
        trial_log.save(run_dir / "trial_log.json")
        print(f"[iter {i}] reward = {reward:.4f}")

        if reward >= cfg.runner.success_threshold:
            print(f"\nSuccess at iteration {i} (reward {reward:.4f} >= threshold {cfg.runner.success_threshold}).")
            break

    best = trial_log.best
    if best:
        print(f"\nBest reward: {best.reward:.4f} at iteration {best.iteration}")
    return trial_log


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


def _setup_run_dir(cfg: Config, resume_log_path: str | None) -> Path:
    """Create (or reuse) the run directory and save the config snapshot."""
    if resume_log_path:
        # Continue in the same directory as the resumed log
        run_dir = Path(resume_log_path).parent
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.runner.log_dir) / f"{cfg.task.name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    # Always save a fresh config snapshot
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(_config_to_dict(cfg), f, indent=2)

    return run_dir


def _load_or_create_log(resume_log_path: str | None) -> TrialLog:
    if not resume_log_path:
        return TrialLog()
    path = Path(resume_log_path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume log not found: {path}")
    log = TrialLog.load(path)
    print(f"Resuming from {path} with {len(log)} existing entries")
    return log


def _config_to_dict(cfg: Config) -> dict:
    """Convert Config dataclass to a JSON-serializable dict."""
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