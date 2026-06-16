"""Closed-loop simulation: controller vs. the true reactor under a disturbance.

Standard control-systems pattern: the controller plans with the fast *surrogate*, but
the *truth* is the Phase 1 ODE plant. We integrate the plant in short zero-order-hold
intervals; each interval the controller measures the (noisy) bed temperature and sets
the actuators.

Scenario: the reactor runs at the yield-optimal setpoint, then at ``dist_time`` a
**cooling failure** drops the jacket capacity. With the safety-feedback controller the
bed is held below the runaway limit; with the controller disabled (setpoints frozen,
no feedback) it runs away. That contrast is the headline result.

Run::

    uv run python -m control.closed_loop            # controlled vs uncontrolled + plot
    uv run python -m control.closed_loop --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.noise import TEMPERATURE
from sim import params as P
from sim.reactor import I_HC, I_T, c5plus_fraction, feed_concentrations, simulate
from .controller import Controller

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def _c5plus_yield(state: np.ndarray, c_co2_in: float) -> float:
    return float(state[I_HC] / c_co2_in) * c5plus_fraction(P.ASF_ALPHA)


def run_closed_loop(
    controller: Controller,
    setpoint: dict[str, float],
    *,
    with_feedback: bool = True,
    dist_time: float = 60.0,
    dist_ua: float = 1.5e4,
    t_end: float = 240.0,
    dt: float = 0.5,
    noise: bool = True,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Step the true plant under control; return logged trajectories.

    The plant is warm-started at the settled operating state for the setpoint, so the
    run isolates the *disturbance response* rather than the startup transient.
    """
    rng = np.random.default_rng(seed)
    ratio, pressure = setpoint["h2_co2_ratio"], setpoint["pressure_bar"]
    catalyst = setpoint["catalyst"]

    op0 = P.OperatingPoint(h2_co2_ratio=ratio, pressure_bar=pressure,
                           coolant_temp=setpoint["coolant_temp"], catalyst=catalyst)
    cfg0 = P.ReactorConfig(ua_per_volume=controller.assumed_ua)
    c_co2_in, _ = feed_concentrations(op0, cfg0)
    # Warm start: integrate to the settled steady state at the nominal setpoint.
    state = simulate(op0, cfg0, t_end=600.0, n_points=50).y[:, -1]
    safety = controller.new_safety(setpoint, dt) if with_feedback else None

    log = {k: [] for k in ("t", "T_C", "yield", "coolant", "ua")}
    t = 0.0
    while t < t_end - 1e-9:
        ua = controller.assumed_ua if t < dist_time else dist_ua
        true_T_C = state[I_T] - 273.15
        meas_T_C = float(TEMPERATURE.read(true_T_C, rng)) if noise else true_T_C

        # Safety feedback regulates bed T via coolant temperature; without it the
        # coolant setpoint is frozen (no disturbance rejection).
        coolant = safety.coolant_temp(meas_T_C) if safety else setpoint["coolant_temp"]

        op = P.OperatingPoint(h2_co2_ratio=ratio, pressure_bar=pressure,
                              coolant_temp=coolant, catalyst=catalyst)
        cfg = P.ReactorConfig(ua_per_volume=ua)
        res = simulate(op, cfg, t_end=dt, n_points=2, y0=state)
        state = res.y[:, -1]
        t += dt

        log["t"].append(t)
        log["T_C"].append(state[I_T] - 273.15)
        log["yield"].append(_c5plus_yield(state, c_co2_in))
        log["coolant"].append(coolant)
        log["ua"].append(ua)

    return {k: np.array(v) for k, v in log.items()}


def plot(controlled: dict, uncontrolled: dict, dist_time: float, *, show: bool) -> Path:
    import matplotlib.pyplot as plt

    limit_C = P.NOMINAL_CONFIG.runaway_temp - 273.15
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(controlled["t"], controlled["T_C"], "tab:blue", label="controlled")
    ax.plot(uncontrolled["t"], uncontrolled["T_C"], "tab:red", label="uncontrolled")
    ax.axhline(limit_C, color="k", ls="--", lw=1, label="runaway limit")
    ax.axvline(dist_time, color="gray", ls=":", lw=1, label="cooling failure")
    ax.set(title="Bed temperature", xlabel="time (s)", ylabel="T (C)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(controlled["t"], np.array(controlled["yield"]) * 100, "tab:blue", label="controlled")
    ax.plot(uncontrolled["t"], np.array(uncontrolled["yield"]) * 100, "tab:red", label="uncontrolled")
    ax.axvline(dist_time, color="gray", ls=":", lw=1)
    ax.set(title="C5+ yield", xlabel="time (s)", ylabel="yield (%)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(controlled["t"], controlled["coolant"], "tab:green", label="controlled")
    ax.plot(uncontrolled["t"], uncontrolled["coolant"], "tab:red", ls="--", label="uncontrolled (frozen)")
    ax.axvline(dist_time, color="gray", ls=":", lw=1, label="cooling failure")
    ax.set(title="Coolant temperature (safety actuator)", xlabel="time (s)",
           ylabel="coolant T (K)")
    ax.legend(fontsize=8)

    fig.suptitle("Closed-loop control vs. cooling-failure disturbance", fontsize=13)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "control_closed_loop.png"
    fig.savefig(out, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out


def _summary(log: dict, label: str) -> str:
    limit_C = P.NOMINAL_CONFIG.runaway_temp - 273.15
    runaway = bool(np.any(log["T_C"] >= limit_C))
    return (f"{label:>13}: T_max={log['T_C'].max():6.1f} C  "
            f"final_yield={log['yield'][-1]*100:5.1f}%  "
            f"[{'RUNAWAY' if runaway else 'safe'}]")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--dist-ua", type=float, default=1.5e4, help="cooling UA after failure")
    parser.add_argument("--no-noise", action="store_true", help="disable sensor noise")
    args = parser.parse_args(argv)

    controller = Controller()
    sp = controller.optimal_setpoint()
    print("RTO optimal setpoint:")
    print(f"  ratio={sp['h2_co2_ratio']:.2f}  pressure={sp['pressure_bar']:.1f} bar  "
          f"coolant={sp['coolant_temp']:.1f} K  catalyst={sp['catalyst']:.2f}")
    print(f"  predicted T={sp['predicted_T_C']:.1f} C, yield={sp['predicted_yield']*100:.1f}%")

    common = dict(dist_ua=args.dist_ua, noise=not args.no_noise)
    controlled = run_closed_loop(controller, sp, with_feedback=True, **common)
    uncontrolled = run_closed_loop(controller, sp, with_feedback=False, **common)

    print(_summary(controlled, "controlled"))
    print(_summary(uncontrolled, "uncontrolled"))

    out = plot(controlled, uncontrolled, dist_time=60.0, show=not args.no_show)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
