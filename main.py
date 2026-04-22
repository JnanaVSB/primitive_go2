"""Entry point for primitive_go2.

    python main.py --config configs/sit.yaml
    python main.py --config configs/sit.yaml --resume logs/sit_20260422_030000/trial_log.json
"""

import argparse
import logging
import sys

from config import load_config
from runner.primitivellm import run


def main():
    parser = argparse.ArgumentParser(description="Run the FORGE loop on the Go2.")
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (e.g., configs/sit.yaml)",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to existing trial_log.json to continue from",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config(args.config)
    run(cfg, resume_log_path=args.resume)


if __name__ == "__main__":
    sys.exit(main())