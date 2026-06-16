"""Design of experiments: sample the reactor's input space and record outputs.

We sweep five inputs with Latin-hypercube sampling (space-filling, far better coverage
than a random or full-factorial grid for the same budget) and run the Phase 1 simulator
at each point, recording the steady-state outputs. The resulting table is:

  * the EDA dataset (analysis/eda.py / eda.ipynb),
  * the input for global sensitivity analysis (analysis/sensitivity.py), and
  * the training set for the Phase 3 PINN surrogate.

Four inputs are the controller's *manipulated* variables; ``cooling_ua`` is a design /
fault parameter (jacket heat-transfer capacity) included because cooling capacity is the
dominant driver of thermal runaway — keeping it in the sweep is what lets us map the
runaway boundary.

Run::

    uv run python -m analysis.sample --n 512 --out analysis/data/doe.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from sim import params as P
from sim.reactor import simulate

# name -> (low, high). Order is fixed and shared with sensitivity.py.
PARAM_SPACE: dict[str, tuple[float, float]] = {
    "h2_co2_ratio": (2.0, 4.0),     # feed H2:CO2 molar ratio
    "pressure_bar": (15.0, 35.0),   # total reactor pressure
    "coolant_temp": (470.0, 540.0), # K, jacket coolant temperature
    "catalyst": (0.5, 3.0),         # catalyst-loading multiplier
    "cooling_ua": (5.0e3, 8.0e4),   # W/(m^3*K), jacket heat-transfer capacity (design/fault)
}
INPUT_NAMES = list(PARAM_SPACE)
OUTPUT_NAMES = ["T_final_C", "T_max_C", "co2_conversion", "c5plus_yield", "runaway"]

DATA_DIR = Path(__file__).resolve().parent / "data"


def run_point(x: np.ndarray, *, t_end: float = 300.0, n_points: int = 200) -> dict[str, float]:
    """Run the simulator at one input vector ``x`` (ordered as ``INPUT_NAMES``)."""
    ratio, pressure, coolant, catalyst, cooling_ua = x
    op = P.OperatingPoint(
        h2_co2_ratio=ratio,
        pressure_bar=pressure,
        coolant_temp=coolant,
        catalyst=catalyst,
    )
    cfg = P.ReactorConfig(ua_per_volume=cooling_ua)
    try:
        res = simulate(op, cfg, t_end=t_end, n_points=n_points)
    except RuntimeError:
        # Integration blow-up counts as an (extreme) runaway, not a usable steady state.
        return {
            "T_final_C": np.nan, "T_max_C": np.nan,
            "co2_conversion": np.nan, "c5plus_yield": np.nan, "runaway": 1.0,
        }
    return {
        "T_final_C": float(res.temperature[-1] - 273.15),
        "T_max_C": float(res.temperature.max() - 273.15),
        "co2_conversion": float(res.co2_conversion()[-1]),
        "c5plus_yield": float(res.c5plus_yield()[-1]),
        "runaway": float(res.runaway),
    }


def sample_inputs(n: int, *, seed: int = 0) -> np.ndarray:
    """Latin-hypercube sample of ``n`` points scaled to ``PARAM_SPACE`` (shape (n, 5))."""
    lows = np.array([lo for lo, _ in PARAM_SPACE.values()])
    highs = np.array([hi for _, hi in PARAM_SPACE.values()])
    unit = qmc.LatinHypercube(d=len(PARAM_SPACE), seed=seed).random(n)
    return qmc.scale(unit, lows, highs)


def build_dataset(n: int, *, seed: int = 0) -> pd.DataFrame:
    """Sample inputs, evaluate the simulator, return a tidy DataFrame."""
    x = sample_inputs(n, seed=seed)
    rows = [{**dict(zip(INPUT_NAMES, xi)), **run_point(xi)} for xi in x]
    return pd.DataFrame(rows, columns=INPUT_NAMES + OUTPUT_NAMES)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512, help="number of LHS samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "doe.csv")
    args = parser.parse_args(argv)

    df = build_dataset(args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n_runaway = int(df["runaway"].sum())
    print(f"wrote {len(df)} samples to {args.out}")
    print(f"  runaway cases : {n_runaway} ({100*n_runaway/len(df):.1f}%)")
    print(f"  C5+ yield     : {df['c5plus_yield'].mean():.3f} mean, "
          f"{df['c5plus_yield'].max():.3f} max")
    print(f"  CO2 conversion: {df['co2_conversion'].mean():.3f} mean")


if __name__ == "__main__":
    main()
