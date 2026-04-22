"""Tests for agent/prompt.py."""

import pytest
from pathlib import Path

from agent.prompt import PromptBuilder, TrialRecord


TEMPLATES_DIR = "templates"


@pytest.fixture
def builder():
    return PromptBuilder(TEMPLATES_DIR)


class TestPromptBuilder:
    def test_templates_dir_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            PromptBuilder("nonexistent_dir_xyz")

    def test_unknown_task_raises(self, builder):
        with pytest.raises(FileNotFoundError):
            builder.build(task="not_a_task", iter_idx=1, max_iters=10)


class TestSitPrompt:
    def test_renders_without_history(self, builder):
        prompt = builder.build(task="sit", iter_idx=1, max_iters=10)
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        # Key content from shared fragments
        assert "Policy" in prompt
        assert "foot_targets" in prompt
        assert "sit" in prompt.lower()
        # Iteration info
        assert "1" in prompt
        assert "10" in prompt

    def test_reward_formula_shown(self, builder):
        prompt = builder.build(task="sit", iter_idx=1, max_iters=10)
        # The reward formula should appear (we agreed to show it)
        assert "R = " in prompt or "reward" in prompt.lower()
        assert "pitch" in prompt.lower()

    def test_history_renders(self, builder):
        history = [
            TrialRecord(
                iteration=1,
                policy_summary="FR=(0.19,-0.27), RR=(-0.05,-0.12), dur=2.5, stiff=normal",
                reward=-0.25,
                rationale="First attempt, based on standard sit geometry.",
            ),
            TrialRecord(
                iteration=2,
                policy_summary="FR=(0.19,-0.27), RR=(-0.10,-0.08), dur=2.5, stiff=normal",
                reward=-0.12,
                rationale="Pulled rear feet in more, pitch improved.",
            ),
        ]
        prompt = builder.build(
            task="sit", iter_idx=3, max_iters=10, trial_history=history,
        )
        assert "Iteration 1" in prompt
        assert "Iteration 2" in prompt
        assert "-0.2500" in prompt
        assert "-0.1200" in prompt
        assert "Pulled rear feet in more" in prompt

    def test_empty_history_renders_cleanly(self, builder):
        prompt = builder.build(task="sit", iter_idx=1, max_iters=10, trial_history=[])
        # Should say something about being iteration 1 / no history
        assert "iteration 1" in prompt.lower() or "no previous" in prompt.lower()


class TestLayPrompt:
    def test_renders(self, builder):
        prompt = builder.build(task="lay", iter_idx=1, max_iters=10)
        assert "lay" in prompt.lower()
        assert "foot_targets" in prompt


class TestIterationVars:
    def test_iter_and_max_substituted(self, builder):
        prompt = builder.build(task="sit", iter_idx=7, max_iters=15)
        assert "7" in prompt
        assert "15" in prompt