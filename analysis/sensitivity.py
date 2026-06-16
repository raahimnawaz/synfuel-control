"""Global sensitivity analysis: which inputs drive yield, temperature, and runaway?

Variance-based (Sobol) sensitivity indices, estimated with the Saltelli sampling
scheme and the Jansen/Saltelli-2010 estimators. Implemented from scratch (no SALib) —
the estimator is short, and it keeps the project dependency-light and self-contained,
matching the from-scratch C++/Rust ports elsewhere in the portfolio.

For each output we report:
  * S1  (first-order index): the fraction of output variance explained by an input
    *on its own*; and
  * ST  (total-effect index): its share including all interactions.

A large ST - S1 gap flags an input that matters mainly through interactions.

Run::

    uv run python -m analysis.sensitivity --n 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from .sample import INPUT_NAMES, PARAM_SPACE, run_point

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
OUTPUTS = ["c5plus_yield", "T_max_C", "runaway"]


def saltelli_matrices(n: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Saltelli matrices A, B (n x D) and AB (D x n x D) over PARAM_SPACE."""
    d = len(PARAM_SPACE)
    lows = np.array([lo for lo, _ in PARAM_SPACE.values()])
    highs = np.array([hi for _, hi in PARAM_SPACE.values()])

    # One low-discrepancy base sample of dimension 2D, split into the A and B halves.
    base = qmc.Sobol(d=2 * d, seed=seed).random(n)
    a = qmc.scale(base[:, :d], lows, highs)
    b = qmc.scale(base[:, d:], lows, highs)

    ab = np.empty((d, n, d))
    for i in range(d):
        ab[i] = a.copy()
        ab[i][:, i] = b[:, i]
    return a, b, ab


def _evaluate(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Run the simulator on every row; return one array per output."""
    records = [run_point(row) for row in matrix]
    out = {k: np.array([r[k] for r in records], dtype=float) for k in OUTPUTS}
    # Sanitise integration blow-ups (rare): cap temperature, zero the yield.
    out["T_max_C"] = np.nan_to_num(out["T_max_C"], nan=1000.0)
    out["c5plus_yield"] = np.nan_to_num(out["c5plus_yield"], nan=0.0)
    return out


def sobol_indices(
    ya: np.ndarray, yb: np.ndarray, yab: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """First-order (S1) and total-effect (ST) indices for one output.

    ya, yb : (n,) outputs on matrices A, B.
    yab    : (D, n) outputs on the AB_i matrices.
    """
    var_y = np.var(np.concatenate([ya, yb]))
    if var_y == 0:
        d = yab.shape[0]
        return np.zeros(d), np.zeros(d)
    s1 = np.mean(yb[None, :] * (yab - ya[None, :]), axis=1) / var_y
    st = 0.5 * np.mean((ya[None, :] - yab) ** 2, axis=1) / var_y
    return s1, st


def analyze(n: int, *, seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Compute S1 and ST for every output. Total model evaluations: n*(D+2)."""
    a, b, ab = saltelli_matrices(n, seed=seed)
    d = a.shape[1]
    fa, fb = _evaluate(a), _evaluate(b)
    fab = [_evaluate(ab[i]) for i in range(d)]

    results: dict[str, dict[str, np.ndarray]] = {}
    for out in OUTPUTS:
        yab = np.vstack([fab[i][out] for i in range(d)])  # (D, n)
        s1, st = sobol_indices(fa[out], fb[out], yab)
        results[out] = {"S1": s1, "ST": st}
    return results


def plot(results: dict[str, dict[str, np.ndarray]], *, show: bool) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(OUTPUTS), figsize=(5 * len(OUTPUTS), 4.5), sharey=True)
    x = np.arange(len(INPUT_NAMES))
    for ax, out in zip(axes, OUTPUTS):
        s1 = np.clip(results[out]["S1"], 0, None)
        st = np.clip(results[out]["ST"], 0, None)
        ax.bar(x - 0.2, s1, 0.4, label="S1 (first-order)", color="tab:blue")
        ax.bar(x + 0.2, st, 0.4, label="ST (total)", color="tab:red", alpha=0.8)
        ax.set_title(out)
        ax.set_xticks(x)
        ax.set_xticklabels(INPUT_NAMES, rotation=40, ha="right", fontsize=8)
    axes[0].set_ylabel("Sobol index")
    axes[0].legend(fontsize=8)
    fig.suptitle("Global sensitivity (Sobol) — what drives each reactor output", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out_path = FIG_DIR / "sensitivity_sobol.png"
    fig.savefig(out_path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256, help="base samples (power of 2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)

    d = len(PARAM_SPACE)
    print(f"running Sobol with n={args.n}  ->  {args.n * (d + 2)} simulator evaluations")
    results = analyze(args.n, seed=args.seed)

    for out in OUTPUTS:
        print(f"\n{out}:")
        order = np.argsort(results[out]["ST"])[::-1]
        for i in order:
            print(f"  {INPUT_NAMES[i]:>14}: S1={results[out]['S1'][i]:+.3f}  "
                  f"ST={results[out]['ST'][i]:+.3f}")

    path = plot(results, show=not args.no_show)
    print(f"\nfigure written to {path}")


if __name__ == "__main__":
    main()
