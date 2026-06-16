"""Analog front-end transfer functions and ADC calibration.

Two sensor channels condition the reactor signals into the ESP32's 0-3.3 V, 12-bit ADC:

  temperature : PT100 RTD divider  ->  non-inverting op-amp gain  ->  RC filter  -> ADC
  pressure    : Wheatstone bridge  ->  instrumentation-amp gain   ->  RC filter  -> ADC

Each stage's maths is derived in circuits/ANALYSIS.md and checked by circuits/verify.py.
The *inverse* calibration (ADC code -> physical units) is what the firmware runs to turn
a raw reading back into degrees C / bar.
"""

from __future__ import annotations

import math

# --- ADC / excitation -----------------------------------------------------------------
VREF = 3.3            # ADC reference & analog supply (V)
ADC_BITS = 12
ADC_MAX = (1 << ADC_BITS) - 1   # 4095

# --- Temperature channel: PT100 RTD divider + non-inverting amp -----------------------
RTD_R0 = 100.0        # PT100 resistance at 0 C (ohms)
RTD_ALPHA = 0.00385   # temperature coefficient (1/C), standard PT100
R_DIVIDER = 1000.0    # top resistor of the divider (ohms)
GAIN_T = 5.0          # non-inverting op-amp gain = 1 + Rf/Rg (Rf=40k, Rg=10k)

# --- Pressure channel: Wheatstone bridge + instrumentation amp ------------------------
P_FULL_SCALE = 50.0   # bar
BRIDGE_SENS = 4.0e-4  # fractional bridge output per bar (Vbridge = VREF*BRIDGE_SENS*P)
GAIN_P = 50.0         # instrumentation-amp gain

# --- RC anti-aliasing filter (shared, one per channel) --------------------------------
RC_R = 10_000.0       # ohms
RC_C = 1.6e-6         # farads
SAMPLE_RATE_HZ = 100.0


def _quantize(volts: float) -> int:
    code = round(max(0.0, min(volts, VREF)) / VREF * ADC_MAX)
    return int(max(0, min(code, ADC_MAX)))


# --- temperature forward / inverse ----------------------------------------------------
def rtd_resistance(temp_C: float) -> float:
    return RTD_R0 * (1.0 + RTD_ALPHA * temp_C)


def temp_divider_voltage(temp_C: float) -> float:
    """Divider node voltage: VREF * R_rtd / (R_divider + R_rtd)."""
    r = rtd_resistance(temp_C)
    return VREF * r / (R_DIVIDER + r)


def temp_frontend_voltage(temp_C: float) -> float:
    """Voltage presented to the ADC after the op-amp gain stage."""
    return GAIN_T * temp_divider_voltage(temp_C)


def temp_to_adc(temp_C: float) -> int:
    return _quantize(temp_frontend_voltage(temp_C))


def adc_to_temp(code: int) -> float:
    """Inverse calibration the firmware runs: ADC code -> degrees C."""
    v_out = code / ADC_MAX * VREF
    v_div = v_out / GAIN_T
    # Invert the divider: R = R_divider * v_div / (VREF - v_div)
    r = R_DIVIDER * v_div / (VREF - v_div)
    return (r / RTD_R0 - 1.0) / RTD_ALPHA


# --- pressure forward / inverse -------------------------------------------------------
def pressure_bridge_voltage(p_bar: float) -> float:
    return VREF * BRIDGE_SENS * p_bar


def pressure_frontend_voltage(p_bar: float) -> float:
    return GAIN_P * pressure_bridge_voltage(p_bar)


def pressure_to_adc(p_bar: float) -> int:
    return _quantize(pressure_frontend_voltage(p_bar))


def adc_to_pressure(code: int) -> float:
    v_out = code / ADC_MAX * VREF
    return v_out / (GAIN_P * VREF * BRIDGE_SENS)


# --- RC filter ------------------------------------------------------------------------
def rc_cutoff_hz() -> float:
    return 1.0 / (2.0 * math.pi * RC_R * RC_C)


def rc_gain_at(freq_hz: float) -> float:
    """First-order low-pass magnitude response |H(f)| = 1/sqrt(1+(f/fc)^2)."""
    return 1.0 / math.sqrt(1.0 + (freq_hz / rc_cutoff_hz()) ** 2)
