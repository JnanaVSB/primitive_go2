"""Tests for agent/parser.py."""

import pytest
import numpy as np

from agent.parser import parse_response, ParseError
from agent.policy import Policy


# Use string concatenation to keep backtick fences from colliding with
# the surrounding triple-quoted string.
FENCE = "```"

CANONICAL_RESPONSE = f"""
Looking at the trial history, the previous policy had the rear feet too
close together. I'll spread them and keep the front legs extended to
emphasize the forward pitch.

{FENCE}python
Policy(
    foot_targets=np.array([
        [ 0.1934, -0.27],
        [ 0.1934, -0.27],
        [-0.12,   -0.05],
        [-0.12,   -0.05],
    ]),
    duration=2.5,
    stiffness='normal',
)
{FENCE}

I expect this to reduce the pitch error.
"""


class TestWellFormedResponses:
    def test_canonical_parses(self):
        policy, _ = parse_response(CANONICAL_RESPONSE)
        assert isinstance(policy, Policy)
        assert policy.duration == 2.5
        assert policy.stiffness == 'normal'
        np.testing.assert_allclose(
            policy.foot_targets,
            [[0.1934, -0.27], [0.1934, -0.27], [-0.12, -0.05], [-0.12, -0.05]],
        )

    def test_rationale_extracted(self):
        _, rationale = parse_response(CANONICAL_RESPONSE)
        assert "trial history" in rationale
        assert "reduce the pitch error" in rationale
        assert "Policy(" not in rationale

    def test_fenced_without_language_tag(self):
        text = (
            "Reasoning.\n\n"
            + FENCE + "\n"
            + "Policy(\n"
            + "    foot_targets=np.array([[0.19, -0.27]] * 4),\n"
            + "    duration=1.5,\n"
            + "    stiffness='soft',\n"
            + ")\n"
            + FENCE + "\n"
        )
        policy, _ = parse_response(text)
        assert policy.duration == 1.5
        assert policy.stiffness == 'soft'


class TestErrorHandling:
    def test_empty_raises(self):
        with pytest.raises(ParseError, match="Empty"):
            parse_response("")

    def test_whitespace_raises(self):
        with pytest.raises(ParseError, match="Empty"):
            parse_response("   \n\t  ")

    def test_no_code_block_raises(self):
        with pytest.raises(ParseError, match="No fenced code block"):
            parse_response("Just some prose, no code here.")

    def test_malformed_python_raises(self):
        text = (
            FENCE + "python\n"
            + "Policy(foot_targets=np.array([[0.1, -0.2] * 4,\n"
            + FENCE + "\n"
        )
        with pytest.raises(ParseError):
            parse_response(text)

    def test_code_without_policy_raises(self):
        text = FENCE + "python\nx = 5\n" + FENCE + "\n"
        with pytest.raises(ParseError):
            parse_response(text)

    def test_wrong_foot_shape_raises(self):
        text = (
            FENCE + "python\n"
            + "Policy(foot_targets=np.array([[0.19, -0.27]]), duration=1.0)\n"
            + FENCE + "\n"
        )
        with pytest.raises(ParseError):
            parse_response(text)

    def test_invalid_stiffness_raises(self):
        text = (
            FENCE + "python\n"
            + "Policy(\n"
            + "    foot_targets=np.array([[0.19, -0.27]] * 4),\n"
            + "    duration=1.0,\n"
            + "    stiffness='medium',\n"
            + ")\n"
            + FENCE + "\n"
        )
        with pytest.raises(ParseError):
            parse_response(text)


class TestSecurity:
    def test_no_imports(self):
        text = FENCE + "python\nimport os\n" + FENCE + "\n"
        with pytest.raises(ParseError):
            parse_response(text)

    def test_no_open_builtin(self):
        text = FENCE + "python\nopen('/etc/passwd')\n" + FENCE + "\n"
        with pytest.raises(ParseError):
            parse_response(text)


class TestVariants:
    def test_assignment_form(self):
        text = (
            FENCE + "python\n"
            + "p = Policy(\n"
            + "    foot_targets=np.array([[0.19, -0.27]] * 4),\n"
            + "    duration=2.0,\n"
            + ")\n"
            + FENCE + "\n"
        )
        policy, _ = parse_response(text)
        assert policy.duration == 2.0

    def test_multiple_blocks_uses_first(self):
        text = (
            "Example from docs:\n\n"
            + FENCE + "python\n"
            + "Policy(foot_targets=np.array([[0.1, -0.1]] * 4), duration=1.0)\n"
            + FENCE + "\n\n"
            + "But actually I want:\n\n"
            + FENCE + "python\n"
            + "Policy(foot_targets=np.array([[0.2, -0.2]] * 4), duration=2.0)\n"
            + FENCE + "\n"
        )
        policy, _ = parse_response(text)
        assert policy.duration == 1.0