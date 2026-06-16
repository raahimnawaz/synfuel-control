"""Tests for the Phase 2 analysis package: DoE sampling, sensor noise, Sobol."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc

from analysis import sensitivity
from analysis.noise import SensorChannel, TEMPERATURE
from analysis.sample import PARAM_SPACE, run_point, sample_inputs


def test_lhs_samples_within_bounds():
    x = sample_inputs(200, seed=1)
    assert x.shape == (200, len(PARAM_SPACE))
    lows = np.array([lo for lo, _ in PARAM_SPACE.values()])
    highs = np.array([hi for _, hi in PARAM_SPACE.values()])
    assert np.all(x >= lows) and np.all(x <= highs)


def test_run_point_outputs_are_physical():
    # A mid-range, well-cooled point.
    x = np.array([3.0, 25.0, 490.0, 1.0, 6.0e4])
    out = run_point(x)
    assert 0.0 <= out["co2_conversion"] <= 1.0
    assert 0.0 <= out["c5plus_yield"] <= 1.0
    assert out["runaway"] in (0.0, 1.0)
    assert out["T_max_C"] >= out["T_final_C"] - 1e-6


def test_sensor_noise_is_deterministic_with_seed():
    a = TEMPERATURE.read(np.full(100, 250.0), np.random.default_rng(7))
    b = TEMPERATURE.read(np.full(100, 250.0), np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_sensor_reading_quantized_when_noise_free():
    # Zero analog noise => reading differs from truth only by quantisation (<= 1 step).
    ch = SensorChannel(name="t", unit="C", phys_min=150.0, phys_max=400.0, noise_std=0.0)
    truth = np.linspace(160.0, 390.0, 50)
    reading = ch.read(truth, np.random.default_rng(0))
    assert np.all(np.abs(reading - truth) <= ch.quantization_step)


def test_sensor_noise_adds_spread():
    truth = np.full(5000, 250.0)
    reading = TEMPERATURE.read(truth, np.random.default_rng(0))
    assert reading.std() > 0.5  # sensor sigma is 1.5 C


def ishigami(x: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * x3**4 * np.sin(x1)


def test_sobol_estimator_recovers_ishigami():
    """Validate the hand-rolled Sobol estimator against the analytic Ishigami indices.

    Analytic (a=7, b=0.1): S1 ~ [0.314, 0.442, 0.0], ST_3 ~ 0.244.
    """
    d, m = 3, 14  # 2**14 base samples
    base = qmc.Sobol(d=2 * d, seed=1).random_base2(m)
    lo, hi = np.full(d, -np.pi), np.full(d, np.pi)
    a_mat, b_mat = qmc.scale(base[:, :d], lo, hi), qmc.scale(base[:, d:], lo, hi)
    ab = np.empty((d, len(a_mat), d))
    for i in range(d):
        ab[i] = a_mat.copy()
        ab[i][:, i] = b_mat[:, i]

    fa, fb = ishigami(a_mat), ishigami(b_mat)
    fab = np.vstack([ishigami(ab[i]) for i in range(d)])
    s1, st = sensitivity.sobol_indices(fa, fb, fab)

    assert s1[0] == pytest.approx(0.314, abs=0.04)
    assert s1[1] == pytest.approx(0.442, abs=0.04)
    assert abs(s1[2]) < 0.04
    assert st[2] == pytest.approx(0.244, abs=0.05)
