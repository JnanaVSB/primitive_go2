"""Trial history (Γ) for the FORGE loop.

Accumulates (iteration, code, reward, rationale) records across iterations
and provides them as TrialRecord objects for prompt rendering. Supports
JSON persistence so runs can be resumed or inspected after the fact.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class TrialEntry:
    """One iteration of the FORGE loop."""
    iteration: int
    code: str
    reward: float
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            'iteration': self.iteration,
            'code': self.code,
            'reward': self.reward,
            'rationale': self.rationale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TrialEntry':
        return cls(
            iteration=d['iteration'],
            code=d['code'],
            reward=d['reward'],
            rationale=d.get('rationale', ''),
        )


class TrialLog:
    """Ordered list of TrialEntry records with persistence."""

    def __init__(self, entries: list[TrialEntry] | None = None):
        self.entries: list[TrialEntry] = list(entries) if entries else []

    def append(self, code: str, reward: float, rationale: str = ""):
        """Add a new entry. Iteration number is inferred from current length."""
        entry = TrialEntry(
            iteration=len(self.entries) + 1,
            code=code,
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
        """Convert entries to the TrialRecord format the prompt builder expects."""
        from agent.prompt import TrialRecord
        records = []
        for e in self.entries:
            # Indent the code for cleaner display in the prompt
            indented_code = "\n".join(
                f"        {line}" for line in e.code.splitlines()
            )
            records.append(TrialRecord(
                iteration=e.iteration,
                policy_summary=indented_code,
                reward=e.reward,
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
>>>>>>> aea4eb5 ( Phase 1: primitives)
