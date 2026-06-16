"""Tests for the Phase 3 surrogate: torch-physics parity, conservation, ONNX, fit."""

from __future__ import annotations

import numpy as np
import torch

from analysis.sample import PARAM_SPACE
from pinn.losses import conservation_residuals, reactor_rhs
from pinn.model import INPUT_DIM, STATE_DIM, ReactorSurrogate
from sim import params as P
from sim.reactor import derivatives, simulate


def _make_model() -> ReactorSurrogate:
    lows = np.array([lo for lo, _ in PARAM_SPACE.values()])
    highs = np.array([hi for _, hi in PARAM_SPACE.values()])
    out_mean = np.array([100.0, 300.0, 5.0, 100.0, 50.0, 520.0])
    out_std = np.array([50.0, 100.0, 5.0, 50.0, 30.0, 40.0])
    return ReactorSurrogate(lows, highs, out_mean, out_std)


def test_model_forward_shape():
    model = _make_model()
    x = torch.zeros(8, INPUT_DIM)
    y = model(x)
    assert y.shape == (8, STATE_DIM)
    assert torch.all(torch.isfinite(y))


def test_torch_rhs_matches_numpy_simulator():
    """The torch physics port must agree with sim.reactor.derivatives."""
    state = np.array([100.0, 320.0, 4.0, 110.0, 48.0, 505.0])
    x = np.array([3.0, 25.0, 490.0, 1.0, 6.0e4])
    op = P.OperatingPoint(h2_co2_ratio=3.0, pressure_bar=25.0,
                          coolant_temp=490.0, catalyst=1.0)
    cfg = P.ReactorConfig(ua_per_volume=6.0e4)

    numpy_rhs = derivatives(0.0, state, op, cfg)
    torch_rhs = reactor_rhs(
        torch.tensor(state[None], dtype=torch.float64),
        torch.tensor(x[None], dtype=torch.float64),
    ).numpy()[0]
    assert np.allclose(numpy_rhs, torch_rhs, rtol=1e-5, atol=1e-5)


def test_conservation_residuals_small_at_true_steady_state():
    """Carbon/hydrogen balances must hold at a real simulator steady state."""
    res = simulate(P.NOMINAL_OP, P.NOMINAL_CONFIG, t_end=400.0)
    state = res.y[:, -1]
    x = np.array([P.NOMINAL_OP.h2_co2_ratio, P.NOMINAL_OP.pressure_bar,
                  P.NOMINAL_OP.coolant_temp, P.NOMINAL_OP.catalyst,
                  P.NOMINAL_CONFIG.ua_per_volume])
    carbon, hydrogen, energy = conservation_residuals(
        torch.tensor(state[None], dtype=torch.float64),
        torch.tensor(x[None], dtype=torch.float64),
    )[0].numpy()
    assert abs(carbon) < 1e-3
    assert abs(hydrogen) < 1e-3
    assert abs(energy) < 5e-2


def test_combined_loss_decreases_with_training():
    """A short training run should reduce the loss (sanity that gradients flow)."""
    from pinn.losses import combined_loss

    torch.manual_seed(0)
    model = _make_model()
    # Tiny synthetic dataset of plausible safe states.
    rng = np.random.default_rng(0)
    lows = model.in_lows.numpy()
    highs = model.in_highs.numpy()
    x = torch.tensor((lows + rng.random((64, INPUT_DIM)) * (highs - lows)),
                     dtype=torch.float32)
    y = model(x).detach() + 0.01 * torch.randn(64, STATE_DIM)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = combined_loss(model, x, y)[0].item()
    for _ in range(200):
        opt.zero_grad()
        loss = combined_loss(model, x, y)[0]
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_onnx_model_runs_if_present():
    """If model.onnx has been exported, it should load and produce finite (B,6) output."""
    from pathlib import Path

    onnx_path = Path(__file__).resolve().parent.parent / "pinn" / "model.onnx"
    if not onnx_path.exists():
        import pytest
        pytest.skip("model.onnx not exported yet (run `python -m pinn.train`)")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probe = np.array([[3.0, 25.0, 490.0, 1.0, 6.0e4]], dtype=np.float32)
    out = sess.run(["state"], {"inputs": probe})[0]
    assert out.shape == (1, STATE_DIM)
    assert np.all(np.isfinite(out))
