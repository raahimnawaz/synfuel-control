# synfuel-control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Closed-loop control of a **simulated Fischer-Tropsch synfuel reactor** — converting
CO₂ + H₂ into liquid hydrocarbons — built as a real-time control system: sense →
estimate → optimise → actuate.

This is fundamentally a **robotics/controls problem** wearing a chemistry hat. The
"plant" is a chemical reactor instead of a motor, but the loop is the same one that runs
on any real machine. The project is **simulated end-to-end (software-in-the-loop)**,
structured so each layer can later drop onto real ESP32 / Jetson Orin Nano hardware.

> **Status: complete (all 6 phases).** First-principles reactor simulator with
> literature-sourced kinetics and a thermal-runaway regime → data-science layer (DoE,
> EDA, Sobol, sensor-noise model) → physics-informed neural surrogate (R² ≈ 0.996, ONNX)
> → closed-loop controller surviving a cooling-failure disturbance → dependency-free C++
> edge engine (~0.98 µs, ~4.8× faster than ONNX Runtime) → analog sensor front-end + ESP32
> firmware closing the full **sense → infer → actuate** loop in software-in-the-loop.

## Architecture (planned)

| Phase | Layer | What it does |
|---|---|---|
| **1** | Reactor sim (`sim/`) | First-principles ODE "plant": coupled RWGS + FT kinetics, energy balance, thermal runaway. ✅ |
| **2** | Data science (`analysis/`) | Latin-hypercube DoE, EDA, from-scratch Sobol sensitivity, sensor/ADC noise model. ✅ |
| **3** | PINN surrogate (`pinn/`) | Physics-informed NN: inputs → steady state, with conservation residuals in the loss; ONNX export. ✅ |
| **4** | Controller (`control/`) | RTO setpoint optimisation + PI safety feedback; rejects a cooling-failure disturbance. ✅ |
| **5** | C++ edge inference (`edge/`) | Hand-rolled dependency-free C++ engine, weights baked in; benchmarked + parity-checked. ✅ |
| **6** | Analog + ESP32 (`circuits/`, `firmware/`, `bridge/`) | Sensor-conditioning circuits, ESP32 firmware, and a software-in-the-loop bridge closing the full loop. ✅ |

## Phase 1 — reactor simulator

A lumped continuous-stirred-tank reactor (CSTR) with two coupled reactions:

```
(1) reverse water-gas-shift   CO2 + H2  <->  CO + H2O      (mildly endothermic)
(2) Fischer-Tropsch growth    CO + 2 H2  ->  (-CH2-) + H2O (strongly exothermic)
```

The Arrhenius temperature dependence of (2) plus its large exothermicity is what
produces **thermal runaway** — hotter → faster → more heat. Keeping the bed below the
runaway limit while maximising the C5+ (liquid-fuel) fraction is the job of the
downstream controller.

Every physical constant (activation energies, reaction enthalpies, ASF chain-growth
probability) is taken from the literature with citations — see
[sim/CHEMISTRY.md](sim/CHEMISTRY.md). Only the rate pre-exponentials are tuned, because
they are catalyst-specific and routinely fitted per reactor.

### Run it

```bash
uv venv --python 3.11
uv pip install numpy scipy matplotlib
uv run python -m sim.run_sim          # simulate + plot stable vs. runaway
```

This writes `figures/reactor_stable_vs_runaway.png` and prints a summary of final
temperature, CO₂ conversion, and C5+ yield for a stable and an aggressive operating
point.

## Phase 2 — data science

Treats the reactor as a system to be characterised statistically:

- **Design of experiments** (`analysis/sample.py`) — Latin-hypercube sweep over five
  inputs (feed ratio, pressure, coolant temperature, catalyst, cooling capacity),
  building the dataset that also trains the Phase 3 surrogate.
- **EDA** (`analysis/eda.py`, `analysis/eda.ipynb`) — runaway boundary, the
  conversion/selectivity trade-off, yield surfaces, and the ASF product distribution.
- **Global sensitivity** (`analysis/sensitivity.py`) — variance-based Sobol indices via
  a **from-scratch Saltelli estimator** (validated against the analytic Ishigami
  function in the tests). Headline finding: **pressure is the dominant driver of thermal
  runaway** (total-effect index ≈ 0.87); cooling capacity matters almost entirely
  through interactions.
- **Sensor + ADC noise model** (`analysis/noise.py`) — 12-bit ESP32 ADC + analog sensor
  noise; the channel is **sensor-limited** (~5.6 ENOB), not ADC-limited. The Phase 6
  firmware reuses this exact model so the software-in-the-loop signal path matches.

```bash
uv run python -m analysis.sample --n 512     # generate the DoE dataset
uv run python -m analysis.eda                # EDA overview figure
uv run python -m analysis.sensitivity --n 256  # Sobol indices + figure
uv run python -m analysis.noise              # sensor-noise demo + characterisation
```

## Phase 3 — physics-informed surrogate

A small MLP maps the five operating inputs to the reactor's **6-dim steady state**
`[CO₂, H₂, CO, H₂O, HC, T]`; conversion, yield, and temperature are derived from that
state. Why predict the full state rather than just yield: it lets the loss enforce the
**governing conservation laws** on the output.

- **`pinn/losses.py`** — the physics-informed loss = data MSE + steady-state
  **conservation residuals** (carbon balance, hydrogen balance, energy balance), written
  as bounded *relative* residuals so they regularise the fit instead of fighting it. (The
  naive `dy/dt = 0` residual is numerically stiff — large production/consumption terms
  cancel — which is why relative balances are used; this is itself a worthwhile
  PINN-engineering lesson.)
- **`pinn/model.py`** — a 64-wide tanh MLP (~5 k params, small enough to hand-roll in
  Phase 5) with input scaling and output de-normalisation baked in as buffers, so the
  exported ONNX graph takes raw physical inputs and returns the physical state.
- Trained on the **safe (non-runaway) branch**, where the input→state map is smooth and
  single-valued. Held-out accuracy: **R² ≈ 0.996** on temperature, conversion, and C5+
  yield (T MAE ≈ 0.7 °C). ONNX output matches torch to ~1e-4.

```bash
uv run python -m pinn.train --n 2500 --epochs 4000   # train, validate, export model.onnx
```

## Phase 4 — closed-loop control

The control layer that makes this a *control* project. Two layers, like a real process
plant:

- **RTO (`controller.py`)** — uses the ONNX surrogate as a fast model to find the
  yield-maximising setpoint subject to a steady-state temperature ceiling, solved by a
  batched search over the (vectorised) surrogate.
- **Safety feedback** — a **PI loop** that regulates bed temperature by adjusting the
  **jacket coolant temperature**, rejecting disturbances the surrogate never saw.

**Headline result** — at the optimal setpoint, a **cooling failure** (jacket capacity
drops ~67%) hits at t = 60 s:

| | controlled | uncontrolled |
|---|---|---|
| peak temperature | **296 °C (safe)** | 328 °C → **runaway** |
| final C5+ yield | 80 % | 89 % (meaningless — catalyst sinters) |

Two engineering lessons are documented in `control/controller.py` and were the crux of
getting this to work:
1. **Coolant temperature, not catalyst, is the effective actuator** — at the operating
   point conversion stays near-complete until the catalyst is almost fully cut, so
   throttling it barely moves temperature; increasing cooling does.
2. **PI, not proportional** — a memoryless proportional law restores the setpoint the
   instant the bed cools and sets up a relaxation oscillation whose peaks can *exceed*
   the uncontrolled runaway. The integral term settles smoothly.

```bash
uv run python -m control.closed_loop          # controlled vs uncontrolled + plot
```

## Phase 5 — C++ edge inference

The trained surrogate, deployed as a **hand-rolled C++ forward pass** with the weights
baked into a generated header (`edge/include/weights.h`) — zero runtime dependencies, so
it drops onto MCU-class / edge hardware (target: Jetson Orin Nano) directly. `python -m
pinn.train` emits both `model.onnx` and the C++ header from the same trained model.

| Engine | Latency / inference | Speedup | Parity vs ONNX |
|---|---|---|---|
| Hand-rolled C++ | **0.98 µs** | **~4.8×** | max abs diff **6.4e-5** (float32 round-off) |
| ONNX Runtime (Python) | 4.71 µs | — | — |

Benchmarked on host; see [edge/bench.md](edge/bench.md) for the honest framing (~1 µs is
the bare inference — the control loop is ms-scale, so inference is never the bottleneck).

```bash
cmake -S edge -B edge/build -DCMAKE_BUILD_TYPE=Release && cmake --build edge/build
./edge/build/synfuel_edge --bench 2000000        # benchmark
echo "3 25 490 1 60000" | ./edge/build/synfuel_edge   # predict a 6-state from stdin
```

## Phase 6 — analog front-end + ESP32 + software-in-the-loop

The embedded layer, which closes the full **sense → infer → actuate** loop.

**Analog front-end (`circuits/`)** — four conditioning stages, each derived by hand in
[circuits/ANALYSIS.md](circuits/ANALYSIS.md), reproduced numerically by
[`circuits/verify.py`](circuits/verify.py), and provided as SPICE decks in
[circuits/spice/](circuits/spice/):

| Stage | Result |
|---|---|
| PT100 RTD divider (temperature) | 0.88 mV/°C; self-heating 1.5 mW |
| Non-inverting op-amp (×5) | 200–330 °C → 2.48–3.05 V; 0.18 °C/code |
| Wheatstone bridge + in-amp (×50) | 50 bar → full-scale 3.3 V |
| RC anti-aliasing (fc = 9.95 Hz) | −15.6 dB at 60 Hz |

(A PT100 RTD is used, not an NTC thermistor — the reactor runs above a thermistor's
~150 °C ceiling.)

**Firmware (`firmware/`)** — `esp32_node.ino` reads the conditioned signals on two ADC
pins, converts to physical units with the *same* calibration as `circuits/frontend.py`,
streams JSON telemetry, and applies coolant-setpoint commands to a PWM valve output.
Runs as-is in **Wokwi** (`diagram.json` + `wokwi.toml`); not compiled in CI (no ESP32
toolchain), but its logic is mirrored and exercised by the SIL below.

**Software-in-the-loop (`bridge/`)** — `run_sil.py` wires every layer together:

```
reactor plant → analog front-end + ADC + noise → virtual ESP32 (JSON) → bridge
   → C++ edge engine (real subprocess) + controller → JSON command → ESP32 → plant
```

A cooling failure hits mid-run; the full distributed loop holds the reactor at **289 °C
(safe)** through it. The deployed **C++ binary serves every in-loop inference** (480
queries, tracking the true steady state to ~4 °C). The JSON protocol is identical to a
real Wokwi/hardware ESP32, so the virtual node swaps for the real one with no bridge
change — the standard **HIL/SIL** development pattern.

```bash
uv run python -m circuits.verify        # check the circuit hand-calcs
uv run python -m bridge.run_sil         # run the full software-in-the-loop demo
```

## License

MIT
