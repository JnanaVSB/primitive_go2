"""Trial history (Γ) for the FORGE loop.

Accumulates (iteration, policies, rewards, rationale) records across iterations
and provides them as TrialRecord objects for prompt rendering. Supports
JSON persistence so runs can be resumed or inspected after the fact.

Handles both single-task runs (one policy, one reward) and sequence runs
(multiple policies, multiple rewards with per-step thresholds).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from agent.policy import Policy


@dataclass
class TrialEntry:
    """One iteration of the FORGE loop.

    For single tasks: policies has 1 item, rewards has 1 item.
    For sequences:    policies has N items, rewards has N items.
    """
    iteration: int
    policies: list[Policy]
    rewards: list[float]
    rationale: str = ""
    step_names: list[str] = field(default_factory=list)

    @property
    def reward(self) -> float:
        """Total reward (sum of all step rewards). Used for best-of ranking."""
        return sum(self.rewards)

    def to_dict(self) -> dict:
        return {
            'iteration': self.iteration,
            'policies': [p.to_dict() for p in self.policies],
            'rewards': self.rewards,
            'rationale': self.rationale,
            'step_names': self.step_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TrialEntry':
        # Backward compatible: old logs have 'policy' and 'reward' (singular)
        if 'policy' in d and 'policies' not in d:
            policies = [Policy.from_dict(d['policy'])]
            rewards = [d['reward']]
            step_names = d.get('step_names', [])
        else:
            policies = [Policy.from_dict(p) for p in d['policies']]
            rewards = d['rewards']
            step_names = d.get('step_names', [])
        return cls(
            iteration=d['iteration'],
            policies=policies,
            rewards=rewards,
            rationale=d.get('rationale', ''),
            step_names=step_names,
        )


class TrialLog:
    """Ordered list of TrialEntry records with persistence."""

    def __init__(self, entries: list[TrialEntry] | None = None):
        self.entries: list[TrialEntry] = list(entries) if entries else []

    def append(
        self,
        policies: list[Policy],
        rewards: list[float],
        rationale: str = "",
        step_names: list[str] | None = None,
    ):
        """Add a new entry. Iteration number is inferred from current length."""
        entry = TrialEntry(
            iteration=len(self.entries) + 1,
            policies=policies,
            rewards=rewards,
            rationale=rationale,
            step_names=step_names or [],
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
        """Entry with the highest total reward, or None if empty."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.reward)

    def to_prompt_records(self) -> list:
        """Convert entries to the TrialRecord format the prompt builder expects."""
        from agent.prompt import TrialRecord
        records = []
        for e in self.entries:
            if len(e.policies) == 1:
                # Single task — same format as before
                policy_summary = _summarize_policy(e.policies[0])
                reward_summary = f"{e.rewards[0]:.4f}"
            else:
                # Sequence — show each step
                parts = []
                for i, (pol, rew) in enumerate(zip(e.policies, e.rewards)):
                    name = e.step_names[i] if i < len(e.step_names) else f"step_{i+1}"
                    parts.append(f"  {name}: {_summarize_policy(pol)}  reward={rew:.4f}")
                policy_summary = "\n".join(parts)
                reward_summary = " | ".join(
                    f"{e.step_names[i] if i < len(e.step_names) else f'step_{i+1}'}: {r:.4f}"
                    for i, r in enumerate(e.rewards)
                )
            records.append(TrialRecord(
                iteration=e.iteration,
                policy_summary=policy_summary,
                reward=reward_summary,
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


def _summarize_policy(policy: Policy) -> str:
    """One-line human-readable summary of a Policy for the prompt history."""
    ft = policy.foot_targets
    parts = []
    for leg, row in zip(['FR', 'FL', 'RR', 'RL'], ft):
        parts.append(f"{leg}=({row[0]:+.3f},{row[1]:+.3f})")
    feet_str = ' '.join(parts)
    return f"{feet_str}  dur={policy.duration}  stiff={policy.stiffness}"