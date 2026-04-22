"""Tests for runner/trial_log.py."""

import json
import pytest
import numpy as np
from pathlib import Path

from agent.policy import Policy
from runner.trial_log import TrialLog, TrialEntry, _summarize_policy


def _make_policy(duration=2.0, stiffness='normal') -> Policy:
    return Policy(
        foot_targets=np.array([[0.19, -0.27]] * 4),
        duration=duration,
        stiffness=stiffness,
    )


class TestTrialLog:
    def test_empty_log(self):
        log = TrialLog()
        assert len(log) == 0
        assert log.best is None
        assert list(log) == []

    def test_append_assigns_iteration(self):
        log = TrialLog()
        log.append(_make_policy(), reward=-0.5, rationale="first try")
        log.append(_make_policy(duration=1.5), reward=-0.3, rationale="second")
        assert len(log) == 2
        assert log[0].iteration == 1
        assert log[1].iteration == 2

    def test_best_returns_highest_reward(self):
        log = TrialLog()
        log.append(_make_policy(), reward=-0.5)
        log.append(_make_policy(), reward=-0.1)
        log.append(_make_policy(), reward=-0.3)
        assert log.best.reward == -0.1
        assert log.best.iteration == 2

    def test_iteration(self):
        log = TrialLog()
        log.append(_make_policy(), reward=-0.5)
        log.append(_make_policy(), reward=-0.3)
        rewards = [e.reward for e in log]
        assert rewards == [-0.5, -0.3]


class TestSerialization:
    def test_roundtrip_save_load(self, tmp_path):
        log = TrialLog()
        log.append(_make_policy(duration=2.0), reward=-0.5, rationale="try A")
        log.append(_make_policy(duration=1.5, stiffness='soft'), reward=-0.2, rationale="try B")

        path = tmp_path / "log.json"
        log.save(path)
        loaded = TrialLog.load(path)

        assert len(loaded) == 2
        assert loaded[0].reward == -0.5
        assert loaded[0].rationale == "try A"
        assert loaded[1].policy.stiffness == 'soft'
        np.testing.assert_allclose(
            loaded[1].policy.foot_targets,
            log[1].policy.foot_targets,
        )

    def test_save_creates_parent_dirs(self, tmp_path):
        log = TrialLog()
        log.append(_make_policy(), reward=-0.1)
        path = tmp_path / "nested" / "subdir" / "log.json"
        log.save(path)
        assert path.exists()

    def test_saved_json_is_readable(self, tmp_path):
        log = TrialLog()
        log.append(_make_policy(), reward=-0.42, rationale="text here")
        path = tmp_path / "log.json"
        log.save(path)

        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]['iteration'] == 1
        assert data[0]['reward'] == -0.42
        assert data[0]['rationale'] == "text here"


class TestPromptRecords:
    def test_converts_to_prompt_format(self):
        log = TrialLog()
        log.append(_make_policy(duration=2.5), reward=-0.3, rationale="reasoning A")

        records = log.to_prompt_records()
        assert len(records) == 1
        r = records[0]
        assert r.iteration == 1
        assert r.reward == -0.3
        assert r.rationale == "reasoning A"
        assert isinstance(r.policy_summary, str)
        assert 'FR' in r.policy_summary
        assert 'dur=2.5' in r.policy_summary

    def test_empty_log_empty_records(self):
        log = TrialLog()
        assert log.to_prompt_records() == []


class TestPolicySummary:
    def test_summary_has_all_legs(self):
        p = _make_policy()
        s = _summarize_policy(p)
        for leg in ('FR', 'FL', 'RR', 'RL'):
            assert leg in s

    def test_summary_compact(self):
        p = _make_policy()
        s = _summarize_policy(p)
        assert '\n' not in s
        assert len(s) < 200