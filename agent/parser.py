"""Parser for LLM policy output.

The LLM returns a text response containing a fenced Python code block with
a Policy(...) expression, plus natural-language reasoning around it.

This module extracts both: the Policy object and the rationale string.
"""

import re
import numpy as np

from agent.policy import Policy


class ParseError(Exception):
    """Raised when the LLM response cannot be parsed into a Policy."""


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)


def parse_response(text: str) -> tuple[Policy, str]:
    """Extract a Policy and rationale from the LLM's text response.

    Args:
        text: full LLM response including code block and prose.

    Returns:
        (policy, rationale) — the Policy object and all non-code text as one string.

    Raises:
        ParseError: no code block found, or code does not produce a valid Policy.
    """
    if not isinstance(text, str) or not text.strip():
        raise ParseError("Empty LLM response")

    code = _extract_code_block(text)
    rationale = _CODE_BLOCK_RE.sub("", text, count=1).strip()
    policy = _exec_policy(code)
    return policy, rationale


def _extract_code_block(text: str) -> str:
    matches = _CODE_BLOCK_RE.findall(text)
    if not matches:
        raise ParseError("No fenced code block found. Expected ```python ... ```.")
    return matches[0].strip()


def _exec_policy(code: str) -> Policy:
    """Execute the code in a restricted namespace and return the Policy.

    The snippet can be a bare Policy(...) expression or statements that
    create a Policy as a variable.
    """
    safe_builtins = {
        "abs": abs, "round": round, "min": min, "max": max,
        "len": len, "range": range, "list": list, "tuple": tuple,
        "float": float, "int": int,
        "True": True, "False": False, "None": None,
    }
    namespace = {"__builtins__": safe_builtins, "Policy": Policy, "np": np}

    try:
        result = eval(code, namespace)
        if isinstance(result, Policy):
            return result
    except SyntaxError:
        pass
    except Exception as e:
        raise ParseError(f"Error evaluating policy: {e}") from e

    try:
        exec(code, namespace)
    except Exception as e:
        raise ParseError(f"Error executing policy code: {e}") from e

    for value in namespace.values():
        if isinstance(value, Policy):
            return value

    raise ParseError("Code did not produce a Policy object.")