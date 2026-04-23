"""Parser for LLM policy output.

The LLM returns a text response containing a fenced Python code block with
either a single Policy(...) expression or a list of Policy(...) expressions,
plus natural-language reasoning around it.

This module extracts both: the Policy object(s) and the rationale string.
"""

import re
import numpy as np

from agent.policy import Policy


class ParseError(Exception):
    """Raised when the LLM response cannot be parsed into a Policy."""


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)


def parse_response(text: str, expected_count: int = 1) -> tuple[list[Policy], str]:
    """Extract Policy/Policies and rationale from the LLM's text response.

    Args:
        text:           full LLM response including code block and prose.
        expected_count: how many policies to expect (1 for single task,
                        2+ for sequences). Used for validation.

    Returns:
        (policies, rationale) — list of Policy objects and all non-code text.

    Raises:
        ParseError: no code block found, or code does not produce valid Policies,
                    or wrong number of policies for a sequence.
    """
    if not isinstance(text, str) or not text.strip():
        raise ParseError("Empty LLM response")

    code = _extract_code_block(text)
    rationale = _CODE_BLOCK_RE.sub("", text).strip()
    policies = _exec_policies(code)

    if len(policies) != expected_count:
        raise ParseError(
            f"Expected {expected_count} policy/policies, got {len(policies)}."
        )

    return policies, rationale


def _extract_code_block(text: str) -> str:
    matches = _CODE_BLOCK_RE.findall(text)
    if not matches:
        raise ParseError("No fenced code block found. Expected ```python ... ```.")
    # Join all code blocks — sequence tasks might use separate blocks per policy
    return "\n".join(m.strip() for m in matches)


def _exec_policies(code: str) -> list[Policy]:
    """Execute the code in a restricted namespace and return all Policies found.

    Handles:
      - A bare Policy(...) expression → [Policy]
      - A list [Policy(...), Policy(...)] → [Policy, Policy]
      - Statements that assign Policy objects to variables → collected in order
    """
    safe_builtins = {
        "abs": abs, "round": round, "min": min, "max": max,
        "len": len, "range": range, "list": list, "tuple": tuple,
        "float": float, "int": int,
        "True": True, "False": False, "None": None,
    }
    namespace = {"__builtins__": safe_builtins, "Policy": Policy, "np": np}

    # Try eval first (handles bare expressions)
    try:
        result = eval(code, namespace)
        if isinstance(result, Policy):
            return [result]
        if isinstance(result, (list, tuple)):
            policies = [x for x in result if isinstance(x, Policy)]
            if policies:
                return policies
    except SyntaxError:
        pass
    except Exception as e:
        raise ParseError(f"Error evaluating policy: {e}") from e

    # Fall back to exec (handles statements and assignments)
    try:
        exec(code, namespace)
    except Exception as e:
        raise ParseError(f"Error executing policy code: {e}") from e

    policies = [v for v in namespace.values() if isinstance(v, Policy)]
    if policies:
        return policies

    raise ParseError("Code did not produce any Policy objects.")