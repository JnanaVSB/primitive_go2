"""Tests for runner/recorder.py."""

import pytest
import numpy as np
from pathlib import Path

from env.env import Go2Env
from runner.recorder import RenderingEnv


XML_PATH = "go2/scene.xml"
HOME_ANGLES = [0.0, 0.9, -1.8] * 4


@pytest.fixture
def wrapped_env():
    base = Go2Env(
        xml_path=XML_PATH,
        control_substeps=4,
        kp=80.0,
        kd=4.0,
        initial_base_height=0.27,
        initial_angles=HOME_ANGLES,
        settle_steps=500,
    )
    base.reset()
    env = RenderingEnv(base, width=320, height=240, fps=30)
    yield env
    env.close()


def test_wrapper_delegates_attributes(wrapped_env):
    """RenderingEnv should pass model, data, qpos access through."""
    assert wrapped_env.model is not None
    assert wrapped_env.data is not None
    assert hasattr(wrapped_env, '_qpos_idx')


def test_frames_captured_on_step(wrapped_env):
    """Each step() appends one frame."""
    assert len(wrapped_env._frames) == 0
    home = np.array(HOME_ANGLES)
    for _ in range(5):
        wrapped_env.step(home)
    assert len(wrapped_env._frames) == 5


def test_reset_clears_frames(wrapped_env):
    """reset() wipes the frame buffer."""
    home = np.array(HOME_ANGLES)
    wrapped_env.step(home)
    wrapped_env.step(home)
    assert len(wrapped_env._frames) == 2
    wrapped_env.reset()
    assert len(wrapped_env._frames) == 0


def test_frame_shape(wrapped_env):
    """Captured frames have the configured dimensions."""
    wrapped_env.step(np.array(HOME_ANGLES))
    frame = wrapped_env._frames[0]
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8


def test_save_video(wrapped_env, tmp_path):
    """save_video() writes a non-empty MP4."""
    home = np.array(HOME_ANGLES)
    for _ in range(30):
        wrapped_env.step(home)
    path = tmp_path / "test.mp4"
    wrapped_env.save_video(path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_save_video_creates_parent_dirs(wrapped_env, tmp_path):
    """Parent directories are created on save."""
    wrapped_env.step(np.array(HOME_ANGLES))
    path = tmp_path / "nested" / "dir" / "test.mp4"
    wrapped_env.save_video(path)
    assert path.exists()


def test_save_video_empty_is_noop(wrapped_env, tmp_path):
    """No frames means no file written (no error)."""
    path = tmp_path / "empty.mp4"
    wrapped_env.save_video(path)
    assert not path.exists()