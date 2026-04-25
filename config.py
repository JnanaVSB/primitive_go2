"""Config loader.

Reads YAML into typed dataclasses. The structure mirrors the YAML files
in configs/ — env, primitive, stiffness_modes, llm, runner, task.

Supports two task formats:
  1. Single task:   task: {name: sit, target: {h: 0.15, ...}}
  2. Sequence:      task: {name: lay_stand, sequence: [{name: lay, target: ...}, {name: stand, target: ...}]}

Each task config repeats the full contents (no inheritance). Simpler
loader, more YAML duplication — accepted tradeoff.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import yaml


@dataclass
class EnvConfig:
    xml_path: str
    control_substeps: int
    initial_base_height: float
    initial_angles: list[float]
    settle_steps: int


@dataclass
class PrimitiveConfig:
    settle_steps_after: int


@dataclass
class StiffnessGains:
    kp: float
    kd: float


@dataclass
class LLMConfig:
    provider: Literal['anthropic', 'openai', 'gemini', 'ollama']
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    base_url: str | None = None
    max_retries: int = 5
    retry_delay: float = 1.0


@dataclass
class RunnerConfig:
    max_iterations: int
    success_threshold: float
    templates_dir: str
    log_dir: str
    max_parse_retries: int = 3


@dataclass
class TargetPose:
    h: float
    roll: float
    pitch: float


@dataclass
class TaskStep:
    """One step in a task sequence (e.g. lay, then stand).

    For gait/walk steps, policy_count > 1 means the LLM outputs multiple
    policies forming one cycle, and loop_duration > 0 means that cycle
    is repeated for that many seconds.
    """
    name: str
    target: TargetPose
    success_threshold: float = -0.05
    distance_weight: float = 0.0
    policy_count: int = 1
    loop_duration: float = 0.0
    phase_duration: float = 5.0


@dataclass
class TaskConfig:
    name: str
    target: TargetPose | None = None
    sequence: list[TaskStep] | None = None

    @property
    def is_sequence(self) -> bool:
        return self.sequence is not None

    @property
    def steps(self) -> list[TaskStep]:
        """Return task steps — works for both single and sequence configs."""
        if self.sequence is not None:
            return self.sequence
        return [TaskStep(name=self.name, target=self.target)]


@dataclass
class Config:
    env: EnvConfig
    primitive: PrimitiveConfig
    stiffness_modes: dict[str, StiffnessGains]
    llm: LLMConfig
    runner: RunnerConfig
    task: TaskConfig


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a Config dataclass.

    Raises:
        FileNotFoundError: config file doesn't exist.
        KeyError: required field missing.
        TypeError: field has wrong type.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    task_raw = raw['task']

    if 'sequence' in task_raw:
        sequence = [
            TaskStep(
                name=step['name'],
                target=TargetPose(**step['target']),
                success_threshold=step.get('success_threshold', -0.05),
                distance_weight=step.get('distance_weight', 0.0),
                policy_count=step.get('policy_count', 1),
                loop_duration=step.get('loop_duration', 0.0),
                phase_duration=step.get('phase_duration', 5.0),
            )
            for step in task_raw['sequence']
        ]
        task = TaskConfig(name=task_raw['name'], sequence=sequence)
    else:
        task = TaskConfig(
            name=task_raw['name'],
            target=TargetPose(**task_raw['target']),
        )

    return Config(
        env=EnvConfig(**raw['env']),
        primitive=PrimitiveConfig(**raw['primitive']),
        stiffness_modes={
            name: StiffnessGains(**gains)
            for name, gains in raw['stiffness_modes'].items()
        },
        llm=LLMConfig(**raw['llm']),
        runner=RunnerConfig(**raw['runner']),
        task=task,
    )