"""Tests for config.py."""

import pytest
from pathlib import Path

from config import (
    load_config, Config, EnvConfig, TaskConfig, LLMConfig, StiffnessGains,
)


CONFIGS_DIR = Path("configs")


class TestLoadConfig:
    def test_sit_config_loads(self):
        cfg = load_config(CONFIGS_DIR / "sit.yaml")
        assert isinstance(cfg, Config)
        assert cfg.task.name == "sit"
        assert cfg.task.target.h == 0.17
        assert cfg.task.target.pitch == -0.3

    def test_lay_config_loads(self):
        cfg = load_config(CONFIGS_DIR / "lay.yaml")
        assert cfg.task.name == "lay"
        assert cfg.task.target.h == 0.10

    def test_env_fields(self):
        cfg = load_config(CONFIGS_DIR / "sit.yaml")
        assert cfg.env.xml_path == "go2/scene.xml"
        assert cfg.env.control_substeps == 4
        assert len(cfg.env.initial_angles) == 12

    def test_stiffness_modes(self):
        cfg = load_config(CONFIGS_DIR / "sit.yaml")
        assert set(cfg.stiffness_modes) == {"soft", "normal", "stiff"}
        assert cfg.stiffness_modes["normal"].kp == 80.0
        assert cfg.stiffness_modes["stiff"].kd == 8.0

    def test_llm_fields(self):
        cfg = load_config(CONFIGS_DIR / "sit.yaml")
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.max_tokens == 2000

    def test_runner_fields(self):
        cfg = load_config(CONFIGS_DIR / "sit.yaml")
        assert cfg.runner.max_iterations == 10
        assert cfg.runner.templates_dir == "templates"


class TestErrors:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/nonexistent.yaml")

    def test_malformed_yaml_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("env: [unclosed")
        with pytest.raises(Exception):
            load_config(bad)

    def test_missing_section_raises(self, tmp_path):
        incomplete = tmp_path / "incomplete.yaml"
        incomplete.write_text("env:\n  xml_path: foo\n")
        with pytest.raises((KeyError, TypeError)):
            load_config(incomplete)