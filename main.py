"""Entry point for primitive_go2.

    python main.py --config configs/sit.yaml
    python main.py --config configs/sit.yaml --resume logs/sit_2026-04-22_05-13-50/trial_log.json
"""

import os
# Use headless GPU rendering (no display needed). Must be set before MuJoCo imports.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import logging
import sys

from config import load_config
from runner.primitivellm import run


def main():
    parser = argparse.ArgumentParser(description="Run the FORGE loop on the Go2.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Console-only base config. Per-run file logging is added later by the loop.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)
    run(cfg, resume_log_path=args.resume)


if __name__ == "__main__":
    sys.exit(main())