"""Train the physics-informed reactor surrogate and export it to ONNX.

Pipeline:
  1. Latin-hypercube sample the input space (reusing the Phase 2 sampler) and integrate
     the Phase 1 simulator to its steady state at each point -> (inputs, state) labels.
  2. Train the MLP with the combined data + physics-residual loss (pinn/losses.py).
  3. Validate on a held-out set: parity of derived temperature / conversion / C5+ yield.
  4. Export model.onnx and check ONNX-vs-torch parity.

Run::

    uv run python -m pinn.train --n 2500 --epochs 4000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from analysis.sample import PARAM_SPACE, sample_inputs
from sim import params as P
from sim.reactor import c5plus_fraction, simulate
from .losses import combined_loss
from .model import INPUT_DIM, ReactorSurrogate

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
ONNX_PATH = HERE / "model.onnx"
WEIGHTS_HEADER = HERE.parent / "edge" / "include" / "weights.h"


# --- data ---------------------------------------------------------------------------
def steady_state(x: np.ndarray) -> np.ndarray | None:
    """Integrate to steady state; return the final 6-state, or None on failure."""
    ratio, pressure, coolant, catalyst, cooling_ua = x
    op = P.OperatingPoint(h2_co2_ratio=ratio, pressure_bar=pressure,
                          coolant_temp=coolant, catalyst=catalyst)
    cfg = P.ReactorConfig(ua_per_volume=cooling_ua)
    try:
        res = simulate(op, cfg, t_end=400.0, n_points=50)
    except RuntimeError:
        return None
    return res.y[:, -1]


def make_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """LHS-sample inputs and integrate each to its steady state.

    We keep only points that settle on the **safe** branch (steady-state temperature
    below the runaway limit). The surrogate is built for the operating region the
    controller actually uses; the hot runaway branch is handled by a temperature
    constraint, not by accurate yield prediction there. Restricting to the safe branch
    also makes the input->state map smooth and single-valued, hence easy to learn.
    """
    xs = sample_inputs(n, seed=seed)
    rows_x, rows_y = [], []
    for x in xs:
        y = steady_state(x)
        if y is not None and np.all(np.isfinite(y)) and y[5] < P.NOMINAL_CONFIG.runaway_temp:
            rows_x.append(x)
            rows_y.append(y)
    return np.array(rows_x), np.array(rows_y)


def feed_co2_in(x: np.ndarray) -> np.ndarray:
    """Inlet CO2 concentration for each input row (for derived metrics)."""
    ratio, pressure = x[:, 0], x[:, 1]
    c_total = (pressure * 1.0e5) / (P.R_GAS * P.NOMINAL_CONFIG.feed_temp)
    return (1.0 / (1.0 + ratio)) * c_total


def derived_metrics(x: np.ndarray, state: np.ndarray) -> dict[str, np.ndarray]:
    """Temperature (C), CO2 conversion, and C5+ yield from raw states."""
    c_co2_in = feed_co2_in(x)
    return {
        "T_C": state[:, 5] - 273.15,
        "conversion": (c_co2_in - state[:, 0]) / c_co2_in,
        "c5plus_yield": (state[:, 4] / c_co2_in) * c5plus_fraction(P.ASF_ALPHA),
    }


# --- training ------------------------------------------------------------------------
def train(
    x_train: np.ndarray, y_train: np.ndarray, *,
    epochs: int, w_phys: float, lr: float, seed: int,
) -> tuple[ReactorSurrogate, dict[str, list[float]]]:
    torch.manual_seed(seed)
    in_lows = np.array([lo for lo, _ in PARAM_SPACE.values()])
    in_highs = np.array([hi for _, hi in PARAM_SPACE.values()])
    out_mean = y_train.mean(axis=0)
    out_std = y_train.std(axis=0) + 1e-6

    model = ReactorSurrogate(in_lows, in_highs, out_mean, out_std)
    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: dict[str, list[float]] = {"total": [], "data": [], "physics": []}
    for ep in range(epochs):
        opt.zero_grad()
        total, data, phys = combined_loss(model, xt, yt, w_phys=w_phys)
        total.backward()
        opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            history["total"].append(total.item())
            history["data"].append(data.item())
            history["physics"].append(phys.item())
    return model, history


# --- plotting / export ---------------------------------------------------------------
def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def validate_and_plot(
    model: ReactorSurrogate, history: dict, x_val: np.ndarray, y_val: np.ndarray,
    *, show: bool,
) -> tuple[Path, dict[str, float]]:
    import matplotlib.pyplot as plt

    with torch.no_grad():
        pred_state = model(torch.as_tensor(x_val, dtype=torch.float32)).numpy()
    true_m = derived_metrics(x_val, y_val)
    pred_m = derived_metrics(x_val, pred_state)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    # Loss curves.
    ax = axes[0]
    epochs_axis = np.arange(len(history["data"])) * 25
    ax.semilogy(epochs_axis, history["data"], label="data")
    ax.semilogy(epochs_axis, history["physics"], label="physics residual")
    ax.set(title="Training loss", xlabel="epoch", ylabel="loss (log)")
    ax.legend(fontsize=8)

    metrics = {}
    panels = [("T_C", "steady-state T (C)"), ("conversion", "CO2 conversion"),
              ("c5plus_yield", "C5+ yield")]
    for ax, (key, label) in zip(axes[1:], panels):
        t, p = true_m[key], pred_m[key]
        r2 = _r2(t, p)
        mae = float(np.mean(np.abs(t - p)))
        metrics[f"{key}_R2"] = r2
        metrics[f"{key}_MAE"] = mae
        ax.scatter(t, p, s=10, alpha=0.5)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(title=f"{label}\nR2={r2:.3f}  MAE={mae:.3g}",
               xlabel="simulator", ylabel="surrogate")

    fig.suptitle("Physics-informed surrogate — validation (held-out)", fontsize=13)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "pinn_validation.png"
    fig.savefig(out, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out, metrics


def export_onnx(model: ReactorSurrogate, path: Path = ONNX_PATH) -> float:
    """Export to ONNX and return the max abs difference vs. torch on random inputs."""
    model.eval()
    dummy = torch.zeros(1, INPUT_DIM, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["inputs"], output_names=["state"],
        dynamic_axes={"inputs": {0: "batch"}, "state": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    # Parity check.
    import onnxruntime as ort

    lows = model.in_lows.numpy()
    highs = model.in_highs.numpy()
    rng = np.random.default_rng(0)
    probe = (lows + rng.random((64, INPUT_DIM)) * (highs - lows)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["state"], {"inputs": probe})[0]
    with torch.no_grad():
        torch_out = model(torch.as_tensor(probe)).numpy()
    return float(np.max(np.abs(onnx_out - torch_out)))


def _c_array(name: str, arr: np.ndarray) -> str:
    """Format a 1-D or 2-D numpy array as a C++ constexpr float initialiser."""
    if arr.ndim == 1:
        body = ", ".join(f"{v:.8e}f" for v in arr)
        return f"constexpr float {name}[{arr.shape[0]}] = {{{body}}};"
    rows = ",\n  ".join(
        "{" + ", ".join(f"{v:.8e}f" for v in row) + "}" for row in arr
    )
    return f"constexpr float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n  {rows}\n}};"


def export_weights_header(model: ReactorSurrogate, path: Path = WEIGHTS_HEADER) -> None:
    """Bake the trained weights + normalisation into a C++ header for the edge port.

    Single source of truth: the same network the ONNX graph encodes, emitted as
    constexpr arrays so the hand-rolled C++ engine has zero runtime dependencies.
    """
    model.eval()
    lin = [model.net[0], model.net[2], model.net[4]]  # the three Linear layers
    h1, h2 = lin[0].out_features, lin[1].out_features
    parts = [
        "// Auto-generated by `python -m pinn.train`. Do not edit by hand.",
        "#pragma once",
        "namespace synfuel {",
        f"constexpr int IN_DIM = {INPUT_DIM};",
        f"constexpr int H1 = {h1};",
        f"constexpr int H2 = {h2};",
        f"constexpr int OUT_DIM = {lin[2].out_features};",
        _c_array("IN_LOWS", model.in_lows.numpy()),
        _c_array("IN_HIGHS", model.in_highs.numpy()),
        _c_array("OUT_MEAN", model.out_mean.numpy()),
        _c_array("OUT_STD", model.out_std.numpy()),
        _c_array("W0", lin[0].weight.detach().numpy()),
        _c_array("B0", lin[0].bias.detach().numpy()),
        _c_array("W1", lin[1].weight.detach().numpy()),
        _c_array("B1", lin[1].bias.detach().numpy()),
        _c_array("W2", lin[2].weight.detach().numpy()),
        _c_array("B2", lin[2].bias.detach().numpy()),
        "}  // namespace synfuel",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2500, help="total samples (LHS)")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--w-phys", type=float, default=0.2, help="physics-loss weight")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)

    print(f"generating {args.n} steady-state samples ...")
    x, y = make_dataset(args.n, seed=args.seed)
    perm = np.random.default_rng(args.seed).permutation(len(x))
    x, y = x[perm], y[perm]
    n_train = int(0.8 * len(x))
    x_tr, y_tr, x_val, y_val = x[:n_train], y[:n_train], x[n_train:], y[n_train:]
    print(f"  usable (safe branch): {len(x)} (train {len(x_tr)}, val {len(x_val)})")

    print(f"training {args.epochs} epochs (w_phys={args.w_phys}) ...")
    model, history = train(x_tr, y_tr, epochs=args.epochs, w_phys=args.w_phys,
                           lr=args.lr, seed=args.seed)
    print(f"  final data loss = {history['data'][-1]:.4e}, "
          f"physics residual = {history['physics'][-1]:.4e}")

    out, metrics = validate_and_plot(model, history, x_val, y_val, show=not args.no_show)
    print("validation (held-out):")
    for key in ("T_C", "conversion", "c5plus_yield"):
        print(f"  {key:>13}: R2={metrics[key+'_R2']:.3f}  MAE={metrics[key+'_MAE']:.4g}")
    print(f"  figure -> {out}")

    max_diff = export_onnx(model)
    print(f"exported {ONNX_PATH}  (ONNX-vs-torch max diff = {max_diff:.2e})")
    export_weights_header(model)
    print(f"exported {WEIGHTS_HEADER}  (C++ edge weights)")


if __name__ == "__main__":
    main()
