"""Rendering wrapper for Go2Env.

Captures frames during step() and saves an MP4 at the end of an iteration.
The base env remains pure — this is an opt-in decorator.

Usage:
    env = Go2Env(...)
    env.reset()
    env = RenderingEnv(env, width=640, height=480)
    # ... run policy ...
    env.save_video('logs/iter_05.mp4')
"""

from pathlib import Path

import imageio
import mujoco
import numpy as np


class RenderingEnv:
    """Wraps a Go2Env to capture frames during step().

    Passes all attribute access through to the underlying env, so code
    that reads env.model, env.data, env._qpos_idx, etc. still works.

    Frame capture runs once per step() call (after the env's control
    substeps), not every physics step — keeps video size manageable.
    """

    def __init__(self, env, width: int = 640, height: int = 480, fps: int = 30):
        self._env = env
        self._renderer = mujoco.Renderer(env.model, height=height, width=width)
        self._frames: list[np.ndarray] = []
        self.fps = fps

    # Delegate attribute access to the wrapped env so primitive.py
    # and kinematics.py work transparently.
    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        result = self._env.step(action)
        self._renderer.update_scene(self._env.data)
        self._frames.append(self._renderer.render().copy())
        return result

    def reset(self, *args, **kwargs):
        self._frames.clear()
        return self._env.reset(*args, **kwargs)

    def save_video(self, path: str | Path):
        """Write captured frames to an MP4."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._frames:
            return
        imageio.mimsave(
            str(path),
            self._frames,
            fps=self.fps,
            codec='libx264',
            quality=7,
        )

    def close(self):
        self._renderer.close()
        if hasattr(self._env, 'close'):
            self._env.close()