"""Numerically verify the analog front-end hand-calculations.

Every number quoted in circuits/ANALYSIS.md is reproduced here from the transfer
functions in circuits/frontend.py. Run as a CLI for a report, or import `checks()` for
the test suite.

    uv run python -m circuits.verify
"""

from __future__ import annotations

from . import frontend as fe


def checks() -> list[tuple[str, float, float, float]]:
    """Return (label, computed, expected, tolerance) rows."""
    rows: list[tuple[str, float, float, float]] = []

    # RTD resistance at reference points.
    rows.append(("PT100 R(0 C) [ohm]", fe.rtd_resistance(0.0), 100.0, 1e-6))
    rows.append(("PT100 R(250 C) [ohm]", fe.rtd_resistance(250.0), 196.25, 1e-2))

    # Divider sensitivity around the operating range (mV/C).
    sens = (fe.temp_divider_voltage(330.0) - fe.temp_divider_voltage(200.0)) / 130.0
    rows.append(("divider sensitivity [mV/C]", sens * 1e3, 0.884, 0.05))

    # Op-amp output must stay within the ADC range across the operating band.
    rows.append(("front-end V at 330 C [V]", fe.temp_frontend_voltage(330.0), 3.053, 0.02))
    rows.append(("front-end V at 200 C [V]", fe.temp_frontend_voltage(200.0), 2.479, 0.02))

    # Temperature round-trip through the 12-bit ADC (quantisation-limited).
    rt = fe.adc_to_temp(fe.temp_to_adc(250.0))
    rows.append(("temp round-trip @250 C [C]", rt, 250.0, 0.3))

    # ADC temperature resolution (degrees per code) near the operating point.
    res = fe.adc_to_temp(fe.temp_to_adc(250.0) + 1) - fe.adc_to_temp(fe.temp_to_adc(250.0))
    rows.append(("ADC resolution [C/code]", res, 0.18, 0.05))

    # Pressure full-scale and round-trip.
    rows.append(("front-end V at 50 bar [V]", fe.pressure_frontend_voltage(50.0), 3.3, 0.02))
    rp = fe.adc_to_pressure(fe.pressure_to_adc(25.0))
    rows.append(("pressure round-trip @25 bar", rp, 25.0, 0.05))

    # RC anti-aliasing filter.
    rows.append(("RC cutoff [Hz]", fe.rc_cutoff_hz(), 9.95, 0.2))
    rows.append(("RC gain @60 Hz", fe.rc_gain_at(60.0), 0.164, 0.01))

    return rows


def main() -> None:
    print(f"ADC: {fe.ADC_BITS}-bit, Vref={fe.VREF} V; sample rate {fe.SAMPLE_RATE_HZ} Hz\n")
    all_ok = True
    for label, got, exp, tol in checks():
        ok = abs(got - exp) <= tol
        all_ok &= ok
        flag = "ok " if ok else "FAIL"
        print(f"  [{flag}] {label:<30} computed={got:10.4f}  expected~{exp:.4f}")
    print("\nall checks passed" if all_ok else "\nSOME CHECKS FAILED")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
