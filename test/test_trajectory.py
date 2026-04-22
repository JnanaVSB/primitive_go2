"""Tests for world/trajectory.py."""

import numpy as np
import pytest

from world.trajectory import quintic_profile, make_trajectory, trajectory_duration_to_nsteps


class TestQuinticProfile:
    def test_endpoints(self):
        """s(0) = 0, s(1) = 1."""
        assert quintic_profile(0.0) == 0.0
        assert quintic_profile(1.0) == 1.0

    def test_midpoint_symmetric(self):
        """s(0.5) should be exactly 0.5 by symmetry."""
        assert abs(quintic_profile(0.5) - 0.5) < 1e-12

    def test_monotonic(self):
        """s(τ) should be monotonically non-decreasing on [0, 1]."""
        taus = np.linspace(0, 1, 101)
        s_values = np.array([quintic_profile(t) for t in taus])
        diffs = np.diff(s_values)
        assert np.all(diffs >= -1e-12), "quintic profile must be non-decreasing"

    def test_zero_velocity_at_endpoints(self):
        """Numerical derivative at τ=0 and τ=1 should be ~0."""
        eps = 1e-4
        # At τ=0
        dv_start = (quintic_profile(eps) - quintic_profile(0.0)) / eps
        assert abs(dv_start) < 1e-4, f"velocity at τ=0 is {dv_start}"
        # At τ=1
        dv_end = (quintic_profile(1.0) - quintic_profile(1.0 - eps)) / eps
        assert abs(dv_end) < 1e-4, f"velocity at τ=1 is {dv_end}"

    def test_clamps_outside_domain(self):
        """τ < 0 returns 0, τ > 1 returns 1."""
        assert quintic_profile(-0.5) == 0.0
        assert quintic_profile(1.5) == 1.0


class TestMakeTrajectory:
    def test_starts_at_start(self):
        start = np.array([0.0, 0.9, -1.8] * 4)
        target = np.array([0.0, 1.2, -1.5] * 4)
        traj = make_trajectory(start, target, duration=2.0)
        np.testing.assert_allclose(traj(0.0), start, atol=1e-12)

    def test_ends_at_target(self):
        start = np.array([0.0, 0.9, -1.8] * 4)
        target = np.array([0.0, 1.2, -1.5] * 4)
        traj = make_trajectory(start, target, duration=2.0)
        np.testing.assert_allclose(traj(2.0), target, atol=1e-12)

    def test_before_start_returns_start(self):
        start = np.array([0.0, 0.9, -1.8] * 4)
        target = np.array([0.0, 1.2, -1.5] * 4)
        traj = make_trajectory(start, target, duration=2.0)
        np.testing.assert_allclose(traj(-1.0), start, atol=1e-12)

    def test_after_end_returns_target(self):
        start = np.array([0.0, 0.9, -1.8] * 4)
        target = np.array([0.0, 1.2, -1.5] * 4)
        traj = make_trajectory(start, target, duration=2.0)
        np.testing.assert_allclose(traj(5.0), target, atol=1e-12)

    def test_midpoint(self):
        """At t = duration/2, joint values = midpoint (symmetric profile)."""
        start = np.array([0.0, 0.9, -1.8] * 4)
        target = np.array([0.4, 1.5, -1.2] * 4)
        traj = make_trajectory(start, target, duration=2.0)
        midpoint = (start + target) / 2
        np.testing.assert_allclose(traj(1.0), midpoint, atol=1e-12)

    def test_shape_preserved(self):
        """Output shape matches input shape."""
        start = np.zeros(12)
        target = np.ones(12)
        traj = make_trajectory(start, target, duration=1.0)
        assert traj(0.5).shape == (12,)

    def test_zero_motion(self):
        """start == target produces a constant trajectory."""
        start = np.array([0.0, 0.9, -1.8] * 4)
        traj = make_trajectory(start, start.copy(), duration=2.0)
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
            np.testing.assert_allclose(traj(t), start, atol=1e-12)

    def test_mismatched_shapes_raises(self):
        with pytest.raises(AssertionError):
            make_trajectory(np.zeros(12), np.zeros(8), duration=1.0)

    def test_nonpositive_duration_raises(self):
        with pytest.raises(AssertionError):
            make_trajectory(np.zeros(12), np.ones(12), duration=0.0)
        with pytest.raises(AssertionError):
            make_trajectory(np.zeros(12), np.ones(12), duration=-1.0)


class TestNSteps:
    def test_exact_division(self):
        assert trajectory_duration_to_nsteps(duration=2.0, dt=0.01) == 200

    def test_rounds_up(self):
        assert trajectory_duration_to_nsteps(duration=2.0, dt=0.015) == 134

    def test_small_duration(self):
        assert trajectory_duration_to_nsteps(duration=0.005, dt=0.01) == 1