"""Exploratory data analysis over the DoE dataset — the 'show the data' deliverable.

Loads analysis/data/doe.csv (from analysis.sample) and renders a six-panel overview:
the runaway boundary, the conversion/selectivity trade-off, yield surfaces, the ASF
product distribution, and output distributions.

Run::

    uv run python -m analysis.sample --n 512        # if the dataset doesn't exist yet
    uv run python -m analysis.eda                   # render figures/eda_overview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sim import params as P
from sim.reactor import asf_weight_fraction

DATA = Path(__file__).resolve().parent / "data" / "doe.csv"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def load(path: Path = DATA) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — generate it first: "
            f"uv run python -m analysis.sample --n 512"
        )
    return pd.read_csv(path)


def overview(df: pd.DataFrame, *, show: bool) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    safe = df[df.runaway == 0]
    runaway = df[df.runaway == 1]

    # 1. Runaway boundary in (pressure, cooling capacity) space.
    ax = axes[0, 0]
    ax.scatter(safe.pressure_bar, safe.cooling_ua, s=14, c="tab:blue", label="safe", alpha=0.7)
    ax.scatter(runaway.pressure_bar, runaway.cooling_ua, s=14, c="tab:red",
               label="runaway", alpha=0.7)
    ax.set(title="Runaway boundary", xlabel="pressure (bar)", ylabel="cooling UA (W/m^3/K)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # 2. Conversion vs C5+ yield trade-off, coloured by peak temperature.
    ax = axes[0, 1]
    sc = ax.scatter(df.co2_conversion, df.c5plus_yield, c=df.T_max_C, cmap="inferno", s=16)
    ax.set(title="Conversion vs C5+ yield", xlabel="CO2 conversion", ylabel="C5+ yield")
    fig.colorbar(sc, ax=ax, label="T_max (C)")

    # 3. C5+ yield surface over (pressure, coolant_temp) for the safe subset.
    ax = axes[0, 2]
    if len(safe) >= 10:
        tc = ax.tricontourf(safe.pressure_bar, safe.coolant_temp, safe.c5plus_yield,
                            levels=12, cmap="viridis")
        fig.colorbar(tc, ax=ax, label="C5+ yield")
    ax.set(title="C5+ yield surface (safe)", xlabel="pressure (bar)", ylabel="coolant T (K)")

    # 4. Anderson-Schulz-Flory product distribution at the model alpha.
    ax = axes[1, 0]
    n = np.arange(1, 21)
    ax.bar(n, asf_weight_fraction(n, P.ASF_ALPHA), color="tab:green")
    ax.axvspan(4.5, 20.5, color="tab:green", alpha=0.12)
    ax.set(title=f"ASF distribution (alpha={P.ASF_ALPHA}); shaded = C5+",
           xlabel="carbon number n", ylabel="mass fraction")

    # 5. Peak-temperature distribution with the runaway limit.
    ax = axes[1, 1]
    ax.hist(df.T_max_C, bins=40, color="tab:purple", alpha=0.8)
    ax.axvline(P.NOMINAL_CONFIG.runaway_temp - 273.15, color="k", ls="--",
               label="runaway limit")
    ax.set(title="Peak temperature", xlabel="T_max (C)", ylabel="count")
    ax.legend(fontsize=8)

    # 6. C5+ yield vs coolant temperature, coloured by runaway.
    ax = axes[1, 2]
    ax.scatter(safe.coolant_temp, safe.c5plus_yield, s=14, c="tab:blue", label="safe", alpha=0.7)
    ax.scatter(runaway.coolant_temp, runaway.c5plus_yield, s=14, c="tab:red",
               label="runaway", alpha=0.7)
    ax.set(title="Yield vs coolant T", xlabel="coolant T (K)", ylabel="C5+ yield")
    ax.legend(fontsize=8)

    fig.suptitle(f"Reactor DoE — exploratory analysis ({len(df)} samples)", fontsize=13)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "eda_overview.png"
    fig.savefig(out, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args(argv)

    df = load(args.data)
    print(f"loaded {len(df)} samples; runaway fraction = {df.runaway.mean():.1%}")
    out = overview(df, show=not args.no_show)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
