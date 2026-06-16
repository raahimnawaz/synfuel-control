"""Phase 6a: analog sensor front-end — transfer functions + calibration.

Single source of truth for the analog signal chain. The ESP32 firmware
(firmware/esp32_node.ino) and the host-side virtual node (bridge/virtual_esp32.py) both
use these same constants and the inverse calibration, so the simulated and (eventual)
real signal paths match.
"""
