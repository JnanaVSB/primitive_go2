"""Sandboxed executor for LLM-generated policy code.

Takes a Python code string from the LLM and runs it with access to:
    - robot: a RobotAPI instance (set_joints, step, get_state)
    - Primitive functions (get_stand_pose, get_sit_pose, etc.)
    - Walk planner API (get_walk_planner, walk_step)
    - numpy as np
    - Basic Python builtins (loops, conditionals, math)

The executor enforces a wall-clock timeout (default 120s) and catches
any exceptions from the LLM's code without crashing the FORGE loop.

Usage:
    result = execute_policy_code(code_string, robot, kinematics)
    if result.success:
        # robot has been moved by the code, read state for reward
    else:
        # result.error has the failure message
"""

import signal
import traceback
from dataclasses import dataclass

import numpy as np

from world.primitives import (
    get_stand_pose,
    get_sit_pose,
    get_lay_pose,
    get_walk_planner,
)
from world.walk_gait import walk_step as _walk_step


@dataclass
class ExecutionResult:
    """Outcome of running LLM-generated code."""
    success: bool
    error: str = ""


class _TimeoutError(Exception):
    """Raised when LLM code exceeds the wall-clock limit."""


def _timeout_handler(signum, frame):
    raise _TimeoutError("Policy code exceeded execution time limit")


def execute_policy_code(
    code: str,
    robot,
    kinematics=None,
    timeout_seconds: int = 120,
) -> ExecutionResult:
    """Execute LLM-generated policy code in a sandboxed namespace.

    Args:
        code:            Python code string from the LLM.
        robot:           RobotAPI instance (already connected to an env).
        kinematics:      Go2Kinematics instance (needed for walk_step).
        timeout_seconds: wall-clock time limit in seconds. Default 120 (2 min).

    Returns:
        ExecutionResult with success=True if the code ran without error,
        or success=False with the error message if it crashed or timed out.
    """
    namespace = _build_namespace(robot, kinematics)

    # Set wall-clock alarm
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        exec(code, namespace)
        return ExecutionResult(success=True)
    except _TimeoutError:
        return ExecutionResult(
            success=False,
            error=f"Timed out after {timeout_seconds} seconds",
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
    finally:
        # Cancel alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _build_namespace(robot, kinematics) -> dict:
    """Build the restricted namespace for LLM code execution."""
    safe_builtins = {
        # Math and type basics
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "float": float,
        "int": int,
        "bool": bool,
        "str": str,
        "True": True,
        "False": False,
        "None": None,
        "print": print,
    }

    # Wrap walk_step so the LLM doesn't need to pass kinematics
    def walk_step(planner, phi):
        """Compute joint angles at cycle phase phi.

        Args:
            planner: a BezierGaitPlanner from get_walk_planner().
            phi:     cycle phase in [0, 1).

        Returns:
            12-dim numpy array of joint angles.
        """
        return _walk_step(planner, phi, kinematics)

    return {
        "__builtins__": safe_builtins,
        # Robot API
        "robot": robot,
        # Static pose primitives
        "get_stand_pose": get_stand_pose,
        "get_sit_pose": get_sit_pose,
        "get_lay_pose": get_lay_pose,
        # Walk planner
        "get_walk_planner": get_walk_planner,
        "walk_step": walk_step,
        # Numpy
        "np": np,
    }