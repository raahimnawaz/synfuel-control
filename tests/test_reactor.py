"""Tests for the Phase 1 reactor model.

Covers the two things most worth guarding: that the ODE conserves atoms (a physics
correctness check), and that the stable/runaway regimes behave as designed.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim import params as P
from sim.reactor import (
    asf_weight_fraction,
    c5plus_fraction,
    feed_concentrations,
    arrhenius,
    simulate,
)
from sim.reactor import I_CO2, I_H2, I_CO, I_H2O, I_HC


def test_feed_concentrations_ratio_and_ideal_gas():
    op = P.OperatingPoint(h2_co2_ratio=3.0, pressure_bar=25.0)
    cfg = P.ReactorConfig()
    c_co2, c_h2 = feed_concentrations(op, cfg)
    # H2:CO2 ratio is respected.
    assert c_h2 / c_co2 == pytest.approx(3.0)
    # Total concentration matches ideal gas P/(RT).
    c_total = (25.0e5) / (P.R_GAS * cfg.feed_temp)
    assert c_co2 + c_h2 == pytest.approx(c_total, rel=1e-12)


def test_arrhenius_increases_with_temperature():
    k_low = arrhenius(P.A_FT, P.EA_FT, 450.0)
    k_high = arrhenius(P.A_FT, P.EA_FT, 550.0)
    assert k_high > k_low > 0.0


def test_asf_distribution_sums_to_one():
    # Mole fractions x_n = (1-a) a^(n-1) sum to 1; weight fractions also sum to 1.
    alpha = P.ASF_ALPHA
    n = np.arange(1, 2000)
    mole = (1 - alpha) * alpha ** (n - 1)
    weight = asf_weight_fraction(n, alpha)
    assert mole.sum() == pytest.approx(1.0, abs=1e-6)
    assert weight.sum() == pytest.approx(1.0, abs=1e-6)


def test_c5plus_fraction_monotonic_in_alpha():
    # Higher chain-growth probability => more heavy (C5+) product.
    assert c5plus_fraction(0.95) > c5plus_fraction(0.90) > c5plus_fraction(0.80)
    assert 0.0 < c5plus_fraction(0.90) < 1.0


def test_carbon_and_hydrogen_conserved_at_steady_state():
    """At steady state the CSTR must conserve atoms fed vs. atoms leaving."""
    res = simulate(P.NOMINAL_OP, P.NOMINAL_CONFIG, t_end=400.0)
    c_co2_in, c_h2_in = feed_concentrations(res.op, res.cfg)
    y_end = res.y[:, -1]

    # Carbon: CO2_in == CO2 + CO + HC (each carries one C).
    carbon_out = y_end[I_CO2] + y_end[I_CO] + y_end[I_HC]
    assert carbon_out == pytest.approx(c_co2_in, rel=1e-3)

    # Hydrogen: H2_in == H2 + H2O + HC (H2O and -CH2- each carry 2 H, like H2).
    hydrogen_out = y_end[I_H2] + y_end[I_H2O] + y_end[I_HC]
    assert hydrogen_out == pytest.approx(c_h2_in, rel=1e-3)


def test_nominal_operating_point_is_stable():
    res = simulate(P.NOMINAL_OP, P.NOMINAL_CONFIG, t_end=300.0)
    assert not res.runaway
    # Settles in the documented cobalt LTFT window with meaningful conversion/yield.
    assert 200.0 < res.temperature[-1] - 273.15 < 260.0
    assert res.co2_conversion()[-1] > 0.2
    assert res.c5plus_yield()[-1] > 0.2


def test_cooling_failure_triggers_runaway():
    cooling_failure = P.ReactorConfig(ua_per_volume=6.0e3)
    res = simulate(P.OperatingPoint(catalyst=1.2), cooling_failure, t_end=300.0)
    assert res.runaway
    assert res.temperature.max() >= res.cfg.runaway_temp
