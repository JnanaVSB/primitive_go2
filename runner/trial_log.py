"""Trial history (Γ) for the FORGE loop.

Accumulates (iteration, policy, reward, rationale) records across iterations
and provides them as TrialRecord objects for prompt rendering. Supports
JSON persistence so runs can be resumed or inspected after the fact.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from agent.policy import Policy


@dataclass
class TrialEntry:
    """One iteration of the FORGE loop.

    This is the internal record. The prompt builder consumes a rendered
    view (TrialRecord) via to_prompt_record(), which flattens the Policy
    into a human-readable summary string.
    """
    iteration: int
    policy: Policy
    reward: float
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            'iteration': self.iteration,
            'policy': self.policy.to_dict(),
            'reward': self.reward,
            'rationale': self.rationale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TrialEntry':
        return cls(
            iteration=d['iteration'],
            policy=Policy.from_dict(d['policy']),
            reward=d['reward'],
            rationale=d.get('rationale', ''),
        )


class TrialLog:
    """Ordered list of TrialEntry records with persistence."""

    def __init__(self, entries: list[TrialEntry] | None = None):
        self.entries: list[TrialEntry] = list(entries) if entries else []

    def append(self, policy: Policy, reward: float, rationale: str = ""):
        """Add a new entry. Iteration number is inferred from current length."""
        entry = TrialEntry(
            iteration=len(self.entries) + 1,
            policy=policy,
            reward=reward,
            rationale=rationale,
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
        """Convert entries to the TrialRecord format the prompt builder expects.

        Imports here to avoid a circular import (prompt.py imports nothing
        from runner; this file imports TrialRecord only when rendering).
        """
        from agent.prompt import TrialRecord
        return [
            TrialRecord(
                iteration=e.iteration,
                policy_summary=_summarize_policy(e.policy),
                reward=e.reward,
                rationale=e.rationale,
            )
            for e in self.entries
        ]

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


def _summarize_policy(policy: Policy) -> str:
    """One-line human-readable summary of a Policy for the prompt history."""
    ft = policy.foot_targets
    parts = []
    for leg, row in zip(['FR', 'FL', 'RR', 'RL'], ft):
        parts.append(f"{leg}=({row[0]:+.3f},{row[1]:+.3f})")
    feet_str = ' '.join(parts)
    return f"{feet_str}  dur={policy.duration}  stiff={policy.stiffness}"