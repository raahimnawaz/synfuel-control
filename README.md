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

> **Status:** Phases 1–2 complete — first-principles reactor simulator with
> literature-sourced kinetics and a demonstrable thermal-runaway regime, plus a
> data-science layer (DoE, EDA, global sensitivity, sensor-noise model).

## Architecture (planned)

| Phase | Layer | What it does |
|---|---|---|
| **1** | Reactor sim (`sim/`) | First-principles ODE "plant": coupled RWGS + FT kinetics, energy balance, thermal runaway. ✅ |
| **2** | Data science (`analysis/`) | Latin-hypercube DoE, EDA, from-scratch Sobol sensitivity, sensor/ADC noise model. ✅ |
| 3 | PINN surrogate (`pinn/`) | Fast NN surrogate with mass/energy balance in the loss. |
| 4 | Controller (`control/`) | Maximise C5+ yield while holding T below the runaway limit. |
| 5 | C++ edge inference (`edge/`) | Compiled inference engine, latency-benchmarked (Jetson target). |
| 6 | Analog + ESP32 (`circuits/`, `firmware/`) | Sensor-conditioning circuits → ESP32 telemetry → actuation. |

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

## License

MIT
