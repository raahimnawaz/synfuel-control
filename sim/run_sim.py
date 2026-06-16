"""CLI: simulate the reactor and plot a trajectory.

Demonstrates two regimes side by side:

  * a STABLE operating point (adequate cooling) that settles to a steady state with
    good C5+ yield, and
  * a THERMAL RUNAWAY (under-cooled / over-catalysed) where the exothermic FT reaction
    outruns heat removal and the bed temperature blows past the safety limit.

Run with::

    uv run python -m sim.run_sim            # show + save figure
    uv run python -m sim.run_sim --no-show  # save only (CI / headless)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from . import params as P
from .reactor import ReactorResult, c5plus_fraction, simulate

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def _summary(result: ReactorResult, label: str) -> str:
    temp = result.temperature
    conv = result.co2_conversion()
    yld = result.c5plus_yield()
    status = "RUNAWAY" if result.runaway else "stable"
    return (
        f"{label:>18}: T_final={temp[-1]-273.15:6.1f} C  "
        f"T_max={temp.max()-273.15:6.1f} C  "
        f"CO2_conv={conv[-1]*100:5.1f}%  "
        f"C5+_yield={yld[-1]*100:5.1f}%  [{status}]"
    )


def plot(stable: ReactorResult, runaway: ReactorResult, *, show: bool) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Fischer-Tropsch synfuel reactor — stable vs. thermal runaway", fontsize=13
    )

    for res, color, name in (
        (stable, "tab:blue", "stable"),
        (runaway, "tab:red", "runaway"),
    ):
        t = res.t
        axes[0, 0].plot(t, res.temperature - 273.15, color=color, label=name)
        axes[0, 1].plot(t, res.co2_conversion() * 100, color=color, label=name)
        axes[1, 0].plot(t, res.c5plus_yield() * 100, color=color, label=name)

    # Temperature with the runaway threshold marked.
    axes[0, 0].axhline(
        P.NOMINAL_CONFIG.runaway_temp - 273.15,
        color="k", ls="--", lw=1, label="runaway limit",
    )
    axes[0, 0].set(title="Bed temperature", xlabel="time (s)", ylabel="T (C)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set(title="CO2 conversion", xlabel="time (s)", ylabel="conversion (%)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set(title="C5+ liquid-fuel yield", xlabel="time (s)", ylabel="yield (%)")
    axes[1, 0].legend(fontsize=8)

    # Species trajectory for the stable run.
    from .reactor import SPECIES  # local import to keep module top tidy

    for i, sp in enumerate(SPECIES):
        axes[1, 1].plot(stable.t, stable.y[i], label=sp)
    axes[1, 1].set(
        title="Species (stable run)", xlabel="time (s)", ylabel="conc. (mol/m^3)"
    )
    axes[1, 1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "reactor_stable_vs_runaway.png"
    fig.savefig(out, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="save figure without displaying")
    parser.add_argument("--t-end", type=float, default=200.0, help="simulation horizon (s)")
    args = parser.parse_args(argv)

    # Stable: nominal cooling.
    stable = simulate(P.NOMINAL_OP, P.NOMINAL_CONFIG, t_end=args.t_end)

    # Runaway: a cooling failure (jacket heat-transfer drops ~10x) lets the exothermic
    # FT reaction outrun heat removal. This is the classic FT runaway trigger; the bed
    # jumps to an unsafe hot branch above the catalyst's safe operating limit.
    cooling_failure = P.ReactorConfig(ua_per_volume=6.0e3)
    hot_op = P.OperatingPoint(catalyst=1.2)
    runaway = simulate(hot_op, cooling_failure, t_end=args.t_end)

    print(f"ASF alpha = {P.ASF_ALPHA}  ->  C5+ mass fraction = {c5plus_fraction():.3f}")
    print(_summary(stable, "stable op"))
    print(_summary(runaway, "aggressive op"))

    out = plot(stable, runaway, show=not args.no_show)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
