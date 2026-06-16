# Analog sensor front-end — circuit analysis

The reactor's physical signals (temperature, pressure) must be conditioned into the
ESP32's **0–3.3 V, 12-bit** ADC window. This document derives each conditioning stage by
hand; every number is reproduced numerically by [`verify.py`](verify.py) and the
transfer functions live in [`frontend.py`](frontend.py) (the single source of truth the
firmware and the SIL both use). SPICE decks for each stage are in [`spice/`](spice/).

Two channels:

```
 temperature:  PT100 RTD divider ──▶ non-inverting op-amp (×5) ──▶ RC low-pass ──▶ ADC
 pressure:     Wheatstone bridge  ──▶ instrumentation amp (×50) ──▶ RC low-pass ──▶ ADC
```

---

## 1. Temperature — PT100 RTD divider

A PT100 resistance-temperature detector (chosen over an NTC thermistor because the
reactor runs at 200–330 °C, far above a thermistor's ~150 °C ceiling):

$$R(T) = R_0\,(1 + \alpha T), \quad R_0 = 100\ \Omega,\ \alpha = 0.00385\ \mathrm{°C^{-1}}$$

So `R(0 °C)=100 Ω`, `R(250 °C)=196.25 Ω`. Placed in a divider with a top resistor
`R_d = 1 kΩ` from the 3.3 V rail:

```
  3.3V ──[ R_d=1k ]──┬── Vdiv ──[ RTD ]── GND
                     └──────────────▶ to op-amp
```

$$V_\text{div}(T) = V_\text{ref}\,\frac{R(T)}{R_d + R(T)}$$

Operating band: `V_div(200 °C)=0.496 V`, `V_div(330 °C)=0.611 V`. **Sensitivity ≈
0.88 mV/°C** — small, hence the gain stage below.

**Self-heating check (why R_d = 1 kΩ):** excitation current `I ≈ 3.3/(1000+200) ≈
2.75 mA`, so RTD dissipation `I²R ≈ 1.5 mW` — negligible self-heating error. A much
smaller `R_d` would raise current and corrupt the reading.

## 2. Temperature — non-inverting op-amp (×5)

The ~0.5 V divider signal is amplified by a non-inverting op-amp:

$$G = 1 + \frac{R_f}{R_g} = 1 + \frac{40\ \mathrm{k}}{10\ \mathrm{k}} = 5$$

Output: `V(200 °C)=2.48 V`, `V(330 °C)=3.05 V` — fits under 3.3 V with margin (ADC
saturates around 390 °C, comfortably above the runaway limit). Inverse calibration the
firmware runs:

$$V_\text{div}=\frac{V_\text{adc}}{G}, \quad R=\frac{R_d\,V_\text{div}}{V_\text{ref}-V_\text{div}}, \quad T=\frac{R/R_0 - 1}{\alpha}$$

ADC temperature resolution ≈ **0.18 °C/code** (12-bit LSB = 0.806 mV ÷ 4.42 mV/°C).

**Honest trade-off:** the divider carries a large DC offset, so only ~2.48–3.05 V of the
0–3.3 V range is used (~17 %). A difference/instrumentation amp would null the offset and
use the full range — which is exactly what the pressure channel does next.

## 3. Pressure — Wheatstone bridge + instrumentation amp (×50)

A piezoresistive pressure sensor as a 4-element bridge gives a small **differential**
output proportional to pressure, with the static offset balanced out:

$$V_\text{bridge}(P) = V_\text{ref}\cdot S \cdot P, \quad S = 4\times10^{-4}\ \mathrm{bar^{-1}}$$

At full scale (50 bar): `V_bridge = 3.3 × 4e-4 × 50 = 66 mV`. An instrumentation amp with
`G = 50` maps that to `3.3 V` — the **full** ADC range (contrast §2). Inverse:
`P = V_adc / (G · V_ref · S)`.

## 4. RC anti-aliasing low-pass filter

Before each ADC, a first-order RC low-pass removes content above the Nyquist frequency
(`f_s/2 = 50 Hz` at our 100 Hz sample rate):

$$f_c = \frac{1}{2\pi R C} = \frac{1}{2\pi (10\,\mathrm{k})(1.6\,\mu\mathrm{F})} = 9.95\ \mathrm{Hz}$$

$$|H(f)| = \frac{1}{\sqrt{1+(f/f_c)^2}}$$

- At 60 Hz (mains hum): `|H| = 0.164` → **−15.6 dB** attenuation.
- Time constant `τ = RC = 16 ms` — far faster than the reactor's seconds-scale dynamics,
  so the filter rejects noise without lagging the real signal.

---

## Summary table (verified by `verify.py`)

| Quantity | Value |
|---|---|
| PT100 R(250 °C) | 196.25 Ω |
| Temp divider sensitivity | 0.88 mV/°C |
| Temp front-end output @330 °C | 3.05 V (< 3.3 V) |
| ADC temperature resolution | 0.18 °C/code |
| Pressure front-end output @50 bar | 3.30 V (full scale) |
| RC cutoff frequency | 9.95 Hz |
| RC attenuation @60 Hz | −15.6 dB |
