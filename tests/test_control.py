"""Tests for the Phase 4 controller: RTO constraint, safety PI, closed-loop behaviour."""

from __future__ import annotations

from control.closed_loop import run_closed_loop
from control.controller import RTO_BOUNDS, Controller, SafetyFeedback
from sim import params as P


def test_optimal_setpoint_respects_temperature_ceiling():
    ctrl = Controller()
    sp = ctrl.optimal_setpoint()
    # Predicted steady-state T must sit at/below the RTO ceiling, with positive yield.
    assert sp["predicted_T_C"] <= ctrl.t_target_C + 0.5
    assert sp["predicted_yield"] > 0.3
    # Setpoint must be inside the actuator bounds.
    for key, (lo, hi) in zip(
        ("h2_co2_ratio", "pressure_bar", "coolant_temp", "catalyst"), RTO_BOUNDS
    ):
        assert lo - 1e-6 <= sp[key] <= hi + 1e-6


def test_safety_feedback_only_cools_when_hot():
    sf = SafetyFeedback(coolant_setpoint=510.0, t_reg_C=274.0, kp=4.0, ki=1.5, dt=0.5)
    # Below the target: no action, coolant stays at the setpoint.
    assert sf.coolant_temp(260.0) == 510.0
    # Above the target: coolant is lowered to remove more heat.
    sf2 = SafetyFeedback(coolant_setpoint=510.0, t_reg_C=274.0, kp=4.0, ki=1.5, dt=0.5)
    assert sf2.coolant_temp(300.0) < 510.0


def test_safety_feedback_clamps_to_minimum():
    sf = SafetyFeedback(coolant_setpoint=510.0, t_reg_C=274.0, kp=4.0, ki=1.5, dt=0.5,
                        coolant_min=440.0)
    # A large, sustained overshoot must not drive the command below the actuator limit.
    out = [sf.coolant_temp(400.0) for _ in range(50)][-1]
    assert out >= 440.0


def test_closed_loop_controlled_safe_uncontrolled_runs_away():
    """The headline result: under a cooling failure, control keeps the bed safe."""
    limit_C = P.NOMINAL_CONFIG.runaway_temp - 273.15
    ctrl = Controller()
    sp = ctrl.optimal_setpoint()
    kw = dict(dist_ua=2.0e4, t_end=140.0, dt=1.0, dist_time=60.0, noise=False)

    controlled = run_closed_loop(ctrl, sp, with_feedback=True, **kw)
    uncontrolled = run_closed_loop(ctrl, sp, with_feedback=False, **kw)

    assert controlled["T_C"].max() < limit_C        # stays safe
    assert uncontrolled["T_C"].max() >= limit_C      # runs away
    # And control preserves a useful yield while doing so.
    assert controlled["yield"][-1] > 0.5
