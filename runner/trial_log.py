"""Trial history (Γ) for the FORGE loop.

Accumulates (iteration, code, reward, per_task_rewards, rationale) records
across iterations. Supports checkpoint-based per-task rewards and JSON
persistence for resume and inspection.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class TrialEntry:
    """One iteration of the FORGE loop."""
    iteration: int
    code: str
    reward: float
    rationale: str = ""
    per_task_rewards: dict | None = None

    def to_dict(self) -> dict:
        d = {
            'iteration': self.iteration,
            'code': self.code,
            'reward': self.reward,
            'rationale': self.rationale,
        }
        if self.per_task_rewards:
            d['per_task_rewards'] = self.per_task_rewards
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'TrialEntry':
        return cls(
            iteration=d['iteration'],
            code=d['code'],
            reward=d['reward'],
            rationale=d.get('rationale', ''),
            per_task_rewards=d.get('per_task_rewards'),
        )


class TrialLog:
    """Ordered list of TrialEntry records with persistence."""

    def __init__(self, entries: list[TrialEntry] | None = None):
        self.entries: list[TrialEntry] = list(entries) if entries else []

    def append(self, code: str, reward: float, rationale: str = "",
               per_task_rewards: dict | None = None):
        """Add a new entry. Iteration number is inferred from current length."""
        entry = TrialEntry(
            iteration=len(self.entries) + 1,
            code=code,
            reward=reward,
            rationale=rationale,
            per_task_rewards=per_task_rewards,
        )
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[TrialEntry]:
        return iter(self.entries)

    def __getitem__(self, idx: int) -> TrialEntry:
        return self.entries[idx]

    @property
    def best(self) -> TrialEntry | None:
        """Entry with the highest reward, or None if empty."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.reward)

    def to_prompt_records(self) -> list:
        """Convert entries to the TrialRecord format the prompt builder expects."""
        from agent.prompt import TrialRecord
        records = []
        for e in self.entries:
            indented_code = "\n".join(
                f"        {line}" for line in e.code.splitlines()
            )

            # Format reward: per-task breakdown if available, otherwise single number
            if e.per_task_rewards:
                reward_str = ", ".join(
                    f"{name}: {r:.4f}" for name, r in e.per_task_rewards.items()
                )
                reward_str += f" | total: {e.reward:.4f}"
            else:
                reward_str = f"{e.reward:.4f}"

            records.append(TrialRecord(
                iteration=e.iteration,
                policy_summary=indented_code,
                reward=reward_str,
                rationale=e.rationale,
            ))
        return records

    def save(self, path: str | Path):
        """Persist to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> 'TrialLog':
        """Load from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls([TrialEntry.from_dict(d) for d in data])