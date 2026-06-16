"""Tests for Phase 6: analog front-end calibration, virtual ESP32, and the SIL loop."""

from __future__ import annotations

import json

from bridge.run_sil import run
from bridge.virtual_esp32 import VirtualESP32
from circuits import frontend as fe
from circuits.verify import checks
from sim import params as P


def test_circuit_handcalcs_match():
    """Every analog front-end hand-calc in ANALYSIS.md must reproduce numerically."""
    for label, got, exp, tol in checks():
        assert abs(got - exp) <= tol, f"{label}: {got} vs {exp} (tol {tol})"


def test_temperature_calibration_roundtrip():
    # The inverse calibration recovers temperature to within ADC quantisation.
    for t_C in (200.0, 250.0, 300.0, 330.0):
        recovered = fe.adc_to_temp(fe.temp_to_adc(t_C))
        assert abs(recovered - t_C) < 0.3


def test_pressure_full_scale_uses_adc_range():
    # 50 bar should land near the top of the ADC range (full-scale design).
    assert fe.pressure_to_adc(50.0) >= 0.98 * fe.ADC_MAX


def test_virtual_esp32_senses_and_actuates():
    node = VirtualESP32(temp_noise_C=0.0, seed=0)
    tele = json.loads(node.telemetry(true_temp_C=260.0, true_press_bar=25.0, t_ms=0))
    assert abs(tele["temp_C"] - 260.0) < 0.5      # noise-free round-trip
    assert abs(tele["press_bar"] - 25.0) < 0.2
    # Command is parsed and clamped to the actuator range.
    assert node.apply_command(json.dumps({"coolant_K": 999.0})) == node.coolant_max
    assert node.apply_command(json.dumps({"coolant_K": 480.0})) == 480.0


def test_sil_loop_stays_safe_end_to_end():
    """The full distributed loop must keep the reactor below the runaway limit."""
    log, mode = run(t_end=140.0, dt=1.0, dist_ua=2.0e4)
    limit_C = P.NOMINAL_CONFIG.runaway_temp - 273.15
    assert log["true_T"].max() < limit_C
    assert log["yield"][-1] > 0.5
    assert len(log["t"]) > 0
