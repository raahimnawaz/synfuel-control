"""Software-in-the-loop: the full distributed control loop, in software.

Wires every layer of the project together:

    reactor plant (Phase 1 ODE)
        -> analog front-end + 12-bit ADC + noise  (Phase 6a / circuits)
        -> virtual ESP32 node, JSON telemetry      (Phase 6b, mirrors the .ino)
        -> bridge
        -> edge inference engine (Phase 5 C++ binary, real subprocess)  [monitor]
           + controller (Phase 4 RTO setpoint + PI safety feedback)
        -> JSON command -> virtual ESP32 -> coolant valve -> plant

A cooling-failure disturbance hits mid-run; the loop must keep the reactor below the
runaway limit. The JSON protocol is identical to what a real Wokwi/hardware ESP32 speaks,
so the virtual node can be swapped for the real one with no change to the bridge.

    uv run python -m bridge.run_sil [--no-show]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from bridge.virtual_esp32 import VirtualESP32
from control.controller import Controller
from sim import params as P
from sim.reactor import I_HC, I_T, c5plus_fraction, feed_concentrations, simulate

_ROOT = Path(__file__).resolve().parent.parent
# The edge binary may be built in-place (clang++) or under edge/build (CMake).
EDGE_BINARY = next(
    (p for p in (_ROOT / "edge" / "synfuel_edge", _ROOT / "edge" / "build" / "synfuel_edge")
     if p.exists()),
    _ROOT / "edge" / "synfuel_edge",
)
FIG_DIR = _ROOT / "figures"


class EdgeInference:
    """Edge inference 'service' — the Phase 5 C++ binary if built, else ONNX in-process.

    Using the compiled binary puts the actual edge engine in the loop over its stdin
    request/response interface (one process, queried each control step).
    """

    def __init__(self, binary: Path = EDGE_BINARY) -> None:
        if binary.exists():
            self.proc = subprocess.Popen(
                [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1,
            )
            self.mode = "C++ binary"
        else:
            from control.controller import Surrogate
            self.surrogate = Surrogate()
            self.proc = None
            self.mode = "ONNX (fallback)"
        self.calls = 0

    def predict_state(self, inputs: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.proc is not None:
            self.proc.stdin.write(" ".join(f"{v:.8g}" for v in inputs) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            return np.array([float(x) for x in line.split()])
        return self.surrogate.state(np.atleast_2d(inputs))[0]

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)


def run(*, dist_time: float = 60.0, dist_ua: float = 2.0e4, t_end: float = 240.0,
        dt: float = 0.5, seed: int = 0) -> tuple[dict, str]:
    controller = Controller()
    sp = controller.optimal_setpoint()
    ratio, pressure, catalyst = sp["h2_co2_ratio"], sp["pressure_bar"], sp["catalyst"]

    op0 = P.OperatingPoint(h2_co2_ratio=ratio, pressure_bar=pressure,
                           coolant_temp=sp["coolant_temp"], catalyst=catalyst)
    cfg0 = P.ReactorConfig(ua_per_volume=controller.assumed_ua)
    c_co2_in, _ = feed_concentrations(op0, cfg0)
    state = simulate(op0, cfg0, t_end=600.0, n_points=50).y[:, -1]  # warm start

    node = VirtualESP32(seed=seed, temp_noise_C=0.4)
    node.applied_coolant = sp["coolant_temp"]
    safety = controller.new_safety(sp, dt)
    edge = EdgeInference()

    log = {k: [] for k in ("t", "true_T", "meas_T", "pred_T", "coolant", "yield")}
    t = 0.0
    try:
        while t < t_end - 1e-9:
            ua = controller.assumed_ua if t < dist_time else dist_ua
            true_T_C = state[I_T] - 273.15

            # ESP32 senses through the analog chain and reports JSON; bridge parses it.
            telemetry = node.telemetry(true_T_C, pressure, int(t * 1000))
            meas_T_C = json.loads(telemetry)["temp_C"]

            # Edge engine (deployed C++ inference) predicts the steady state for the
            # current operating conditions -- a model-in-the-loop monitor. Given a
            # cooling-health estimate it tracks the true steady state, validating the
            # surrogate live in the loop.
            pred_state = edge.predict_state(
                np.array([ratio, pressure, node.applied_coolant, catalyst, ua])
            )
            pred_T_C = pred_state[5] - 273.15

            # Controller computes the coolant command from the measurement; ESP32 applies.
            coolant_cmd = safety.coolant_temp(meas_T_C)
            applied = node.apply_command(json.dumps({"coolant_K": coolant_cmd}))

            op = P.OperatingPoint(h2_co2_ratio=ratio, pressure_bar=pressure,
                                  coolant_temp=applied, catalyst=catalyst)
            state = simulate(op, P.ReactorConfig(ua_per_volume=ua), t_end=dt,
                             n_points=2, y0=state).y[:, -1]
            t += dt

            log["t"].append(t)
            log["true_T"].append(state[I_T] - 273.15)
            log["meas_T"].append(meas_T_C)
            log["pred_T"].append(pred_T_C)
            log["coolant"].append(applied)
            log["yield"].append(float(state[I_HC] / c_co2_in) * c5plus_fraction(P.ASF_ALPHA))
    finally:
        edge.close()

    return {k: np.array(v) for k, v in log.items()}, edge.mode


def plot(log: dict, *, show: bool) -> Path:
    import matplotlib.pyplot as plt

    limit_C = P.NOMINAL_CONFIG.runaway_temp - 273.15
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(log["t"], log["true_T"], "tab:blue", label="true (plant)")
    ax.plot(log["t"], log["meas_T"], "tab:orange", lw=0.7, alpha=0.85,
            label="measured (RTD → op-amp → ADC → ESP32)")
    ax.axhline(limit_C, color="k", ls="--", lw=1, label="runaway limit")
    ax.axvline(60.0, color="gray", ls=":", lw=1, label="cooling failure")
    ax.set(title="Bed temperature through the full loop", xlabel="time (s)", ylabel="T (C)")
    ax.legend(fontsize=8)

    axes[1].plot(log["t"], log["coolant"], "tab:purple")
    axes[1].axvline(60.0, color="gray", ls=":", lw=1)
    axes[1].set(title="Coolant command (bridge -> ESP32)", xlabel="time (s)",
                ylabel="coolant T (K)")

    axes[2].plot(log["t"], np.array(log["yield"]) * 100, "tab:blue")
    axes[2].axvline(60.0, color="gray", ls=":", lw=1)
    axes[2].set(title="C5+ yield", xlabel="time (s)", ylabel="yield (%)")

    fig.suptitle("Software-in-the-loop: plant -> analog front-end -> ESP32 -> bridge "
                 "-> edge inference + controller -> command", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "sil_closed_loop.png"
    fig.savefig(out, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)

    log, mode = run()
    limit_C = P.NOMINAL_CONFIG.runaway_temp - 273.15
    safe = bool(log["true_T"].max() < limit_C)
    # Edge-monitor accuracy in the loop: mean |predicted - true| once settled (t > 120 s).
    settled = log["t"] > 120.0
    residual = float(np.mean(np.abs(log["pred_T"][settled] - log["true_T"][settled])))
    print(f"edge inference engine : {mode}  ({len(log['t'])} in-loop queries)")
    print(f"edge-monitor tracking : {residual:.1f} C mean |predicted - true| (settled)")
    print(f"full-loop result      : T_max={log['true_T'].max():.1f} C  "
          f"final_yield={log['yield'][-1]*100:.1f}%  [{'safe' if safe else 'RUNAWAY'}]")
    out = plot(log, show=not args.no_show)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
