"""The controller: surrogate-based setpoint optimisation + safety feedback.

Two layers, exactly as a real process-control stack is organised:

  * **RTO (real-time optimisation)** — uses the Phase 3 ONNX surrogate as a fast model
    to solve a constrained optimisation: maximise C5+ yield subject to the steady-state
    temperature staying below a safe operating ceiling, within actuator bounds. Slow.

  * **Safety feedback** — a fast PI loop that regulates the bed temperature by adjusting
    the **jacket coolant temperature** (the effective heat-removal lever). When the bed
    runs hot, it lowers the coolant temperature to remove more heat. This is what keeps
    the plant safe under disturbances the surrogate never saw (e.g. a cooling failure).

Two design lessons baked in here (both worth being able to explain):
  * *coolant temperature, not catalyst, is the effective actuator* — at this operating
    point conversion stays near-complete until the catalyst is almost fully cut, so
    throttling catalyst barely moves the temperature; increasing cooling does;
  * *PI, not proportional* — a memoryless proportional law restores the setpoint the
    instant the bed dips below target and sets up a relaxation oscillation whose peaks
    can exceed the uncontrolled runaway; the integral term settles smoothly instead.

The RTO operates with a margin below the runaway limit precisely so the feedback layer
has room to reject disturbances the surrogate cannot anticipate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from sim import params as P
from sim.reactor import c5plus_fraction

ONNX_PATH = Path(__file__).resolve().parent.parent / "pinn" / "model.onnx"

# Manipulated-variable order for the optimiser: ratio, pressure, coolant_temp, catalyst.
RTO_BOUNDS = [(2.0, 4.0), (15.0, 35.0), (470.0, 540.0), (0.5, 3.0)]


class Surrogate:
    """Thin wrapper around the exported ONNX reactor surrogate."""

    def __init__(self, path: Path = ONNX_PATH) -> None:
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def state(self, inputs: np.ndarray) -> np.ndarray:
        """Predict the steady state for inputs of shape (N, 5) (or (5,))."""
        x = np.atleast_2d(inputs).astype(np.float32)
        return self.sess.run(["state"], {"inputs": x})[0]

    def metrics(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        """Derived steady-state metrics: temperature (C), conversion, C5+ yield."""
        x = np.atleast_2d(inputs).astype(np.float64)
        st = self.state(x)
        c_total = (x[:, 1] * 1.0e5) / (P.R_GAS * P.NOMINAL_CONFIG.feed_temp)
        c_co2_in = (1.0 / (1.0 + x[:, 0])) * c_total
        return {
            "T_C": st[:, 5] - 273.15,
            "conversion": (c_co2_in - st[:, 0]) / c_co2_in,
            "c5plus_yield": (st[:, 4] / c_co2_in) * c5plus_fraction(P.ASF_ALPHA),
        }


class SafetyFeedback:
    """Stateful PI loop regulating bed temperature via the coolant temperature.

    Lowers the jacket coolant temperature (within limits) when the measured bed
    temperature exceeds ``t_reg_C``. Integral state carries across control steps; an
    anti-windup clamp bounds it to the actuator's authority. Only ever *increases*
    cooling — in normal operation it sits at the coolant setpoint and does nothing.
    """

    def __init__(
        self, coolant_setpoint: float, t_reg_C: float, kp: float, ki: float, dt: float,
        coolant_min: float = 440.0,
    ) -> None:
        self.coolant_setpoint = coolant_setpoint
        self.t_reg_C = t_reg_C
        self.kp = kp
        self.ki = ki
        self.dt = dt
        self.coolant_min = coolant_min
        self.integral = 0.0
        self._i_max = (coolant_setpoint - coolant_min) / max(ki, 1e-9)

    def coolant_temp(self, measured_T_C: float) -> float:
        """Return the commanded coolant temperature for this control step."""
        error = measured_T_C - self.t_reg_C  # positive => too hot
        self.integral = float(np.clip(self.integral + error * self.dt, 0.0, self._i_max))
        drop = max(0.0, self.kp * error + self.ki * self.integral)
        return float(np.clip(self.coolant_setpoint - drop, self.coolant_min,
                             self.coolant_setpoint))


@dataclass
class Controller:
    """Two-layer controller around a surrogate of the reactor."""

    surrogate: Surrogate = field(default_factory=Surrogate)
    assumed_ua: float = P.NOMINAL_CONFIG.ua_per_volume  # cooling the RTO assumes
    t_target_C: float = 270.0        # RTO operating ceiling (leaves disturbance margin)
    # Safety PI gains (regulating bed T via coolant temperature).
    t_reg_C: float = 274.0           # regulation target for the safety loop
    kp: float = 4.0                  # K of coolant drop per degree of error
    ki: float = 1.5                  # integral gain
    coolant_min: float = 440.0

    def new_safety(self, setpoint: dict[str, float], dt: float) -> SafetyFeedback:
        """Create a fresh safety-feedback loop bound to a setpoint's coolant level."""
        return SafetyFeedback(setpoint["coolant_temp"], self.t_reg_C, self.kp, self.ki,
                              dt, self.coolant_min)

    def optimal_setpoint(self, *, n_samples: int = 20000, seed: int = 0) -> dict[str, float]:
        """Maximise C5+ yield s.t. steady-state T <= t_safe, within actuator bounds.

        Solved by evaluating the (vectorised) surrogate on a large quasi-random sample of
        the manipulated-variable space and picking the highest-yield feasible point. This
        is fast (one batched ONNX call) and robust — no gradient noise from the float32
        model, and it cannot get stuck in a local optimum the way SLSQP did.
        """
        lows = np.array([lo for lo, _ in RTO_BOUNDS])
        highs = np.array([hi for _, hi in RTO_BOUNDS])
        rng = np.random.default_rng(seed)
        u = lows + rng.random((n_samples, 4)) * (highs - lows)
        full = np.column_stack([u, np.full(len(u), self.assumed_ua)])

        m = self.surrogate.metrics(full)
        feasible = m["T_C"] <= self.t_target_C
        if not feasible.any():
            raise RuntimeError("no feasible operating point under the temperature limit")
        scored = np.where(feasible, m["c5plus_yield"], -np.inf)
        best = int(np.argmax(scored))
        return {
            "h2_co2_ratio": float(u[best, 0]), "pressure_bar": float(u[best, 1]),
            "coolant_temp": float(u[best, 2]), "catalyst": float(u[best, 3]),
            "predicted_T_C": float(m["T_C"][best]),
            "predicted_yield": float(m["c5plus_yield"][best]),
        }
