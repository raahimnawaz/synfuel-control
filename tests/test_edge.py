"""Tests for the Phase 5 C++ edge engine: builds it and checks parity with ONNX."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
EDGE = ROOT / "edge"


def _compiler() -> str | None:
    for cc in ("clang++", "g++"):
        if shutil.which(cc):
            return cc
    return None


@pytest.fixture(scope="module")
def edge_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cc = _compiler()
    if cc is None:
        pytest.skip("no C++ compiler available")
    out = tmp_path_factory.mktemp("edge") / "synfuel_edge"
    subprocess.run(
        [cc, "-std=c++17", "-O2", "-I", str(EDGE / "include"),
         str(EDGE / "src" / "main.cpp"), str(EDGE / "src" / "mlp.cpp"), "-o", str(out)],
        check=True,
    )
    return out


def test_cpp_matches_onnx(edge_binary: Path):
    """The hand-rolled C++ engine must reproduce the ONNX model to ~float32 precision."""
    if not (ROOT / "pinn" / "model.onnx").exists():
        pytest.skip("model.onnx not exported yet")
    import onnxruntime as ort

    rng = np.random.default_rng(123)
    lo = np.array([2.0, 15.0, 470.0, 0.5, 5e3])
    hi = np.array([4.0, 35.0, 540.0, 3.0, 8e4])
    probe = (lo + rng.random((100, 5)) * (hi - lo)).astype(np.float32)

    sess = ort.InferenceSession(str(ROOT / "pinn" / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["state"], {"inputs": probe})[0]

    stdin = "\n".join(" ".join(f"{v:.8g}" for v in row) for row in probe)
    res = subprocess.run([str(edge_binary)], input=stdin, capture_output=True,
                         text=True, check=True)
    cpp_out = np.array([[float(x) for x in line.split()]
                        for line in res.stdout.strip().splitlines()])

    assert cpp_out.shape == onnx_out.shape
    assert np.max(np.abs(cpp_out - onnx_out)) < 1e-3


def test_bench_runs(edge_binary: Path):
    res = subprocess.run([str(edge_binary), "--bench", "10000"],
                         capture_output=True, text=True, check=True)
    assert "mean latency" in res.stdout
