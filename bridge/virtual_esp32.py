"""A host-side virtual ESP32 node — mirrors firmware/esp32_node.ino.

It runs the *same* analog front-end forward path + ADC quantisation (circuits/frontend.py)
and the *same* inverse calibration the firmware uses, so the measurement the bridge sees
is produced exactly as it would be on hardware. Telemetry and commands use the identical
JSON protocol, so this node can be swapped for a real Wokwi/ESP32 over a socket with no
change to the bridge.
"""

from __future__ import annotations

import json

import numpy as np

from circuits import frontend as fe


class VirtualESP32:
    """Stands in for the ESP32: senses (via the analog chain) and actuates."""

    def __init__(self, *, temp_noise_C: float = 1.0, seed: int = 0,
                 coolant_min: float = 440.0, coolant_max: float = 540.0) -> None:
        self.temp_noise_C = temp_noise_C
        self.rng = np.random.default_rng(seed)
        self.coolant_min = coolant_min
        self.coolant_max = coolant_max
        self.applied_coolant = coolant_max  # last commanded coolant temperature

    def telemetry(self, true_temp_C: float, true_press_bar: float, t_ms: int) -> str:
        """Sense through the analog front-end + 12-bit ADC; emit a JSON line."""
        # Sensor noise enters in the physical domain, then the circuit + ADC discretise it.
        noisy_T = true_temp_C + self.rng.normal(0.0, self.temp_noise_C)
        temp_code = fe.temp_to_adc(noisy_T)
        press_code = fe.pressure_to_adc(true_press_bar)
        return json.dumps({
            "t_ms": t_ms,
            "temp_C": round(fe.adc_to_temp(temp_code), 2),
            "press_bar": round(fe.adc_to_pressure(press_code), 2),
        })

    def apply_command(self, command_json: str) -> float:
        """Parse a JSON command and 'apply' the coolant setpoint; return applied value."""
        try:
            cmd = json.loads(command_json)
        except json.JSONDecodeError:
            return self.applied_coolant
        if "coolant_K" in cmd:
            self.applied_coolant = float(
                np.clip(cmd["coolant_K"], self.coolant_min, self.coolant_max)
            )
        return self.applied_coolant
