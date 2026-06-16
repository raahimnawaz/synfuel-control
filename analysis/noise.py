"""Realistic sensor + ADC noise model for the reactor telemetry.

Phase 1 gives clean physics. Real telemetry is not clean: a sensor adds analog noise,
and the ESP32's 12-bit ADC quantises the conditioned voltage. This module turns a clean
physical signal into the messy reading the controller will actually see, so the control
side is designed and tested against realistic data.

It is deliberately the *single source of truth* for the noise characteristics: the
Phase 6 ESP32 firmware reuses the same `SensorChannel` definitions, so the
software-in-the-loop signal path matches the simulated one.

Signal path modelled (matches the Phase 6 analog front-end):

    physical value --(+ Gaussian sensor noise)--> sensor volts in [0, Vref]
                   --(quantise to ADC LSB)-------> ADC code
                   --(scale back)----------------> reading in physical units

Run a demo::

    uv run python -m analysis.noise            # corrupt a reactor trajectory + plot
    uv run python -m analysis.noise --no-show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class SensorChannel:
    """A sensor + ADC channel mapping a physical range onto a digitised reading.

    Parameters
    ----------
    name : label, e.g. "temperature".
    unit : physical unit, e.g. "C".
    phys_min, phys_max : full-scale physical range the sensor maps onto [0, Vref].
    noise_std : 1-sigma analog sensor noise, in *physical* units.
    adc_bits : ADC resolution (ESP32 SAR ADC is 12-bit).
    vref : ADC reference voltage (ESP32 ~ 3.3 V).
    """

    name: str
    unit: str
    phys_min: float
    phys_max: float
    noise_std: float
    adc_bits: int = 12
    vref: float = 3.3

    @property
    def levels(self) -> int:
        return 2 ** self.adc_bits

    @property
    def lsb_volts(self) -> float:
        """Voltage represented by one ADC count."""
        return self.vref / self.levels

    @property
    def quantization_step(self) -> float:
        """ADC resolution expressed back in physical units."""
        return (self.phys_max - self.phys_min) / self.levels

    def to_volts(self, phys: np.ndarray | float) -> np.ndarray | float:
        span = self.phys_max - self.phys_min
        v = (np.asarray(phys, dtype=float) - self.phys_min) / span * self.vref
        return np.clip(v, 0.0, self.vref)

    def from_volts(self, volts: np.ndarray | float) -> np.ndarray | float:
        span = self.phys_max - self.phys_min
        return self.phys_min + np.asarray(volts, dtype=float) / self.vref * span

    def quantize(self, volts: np.ndarray | float) -> np.ndarray | float:
        code = np.round(np.asarray(volts, dtype=float) / self.lsb_volts)
        code = np.clip(code, 0, self.levels - 1)
        return code * self.lsb_volts

    def read(
        self, phys: np.ndarray | float, rng: np.random.Generator
    ) -> np.ndarray | float:
        """Full path: add analog noise, convert to volts, quantise, convert back."""
        phys = np.asarray(phys, dtype=float)
        noisy = phys + rng.normal(0.0, self.noise_std, size=phys.shape)
        return self.from_volts(self.quantize(self.to_volts(noisy)))


# Pre-configured channels matching the reactor's physical ranges. Phase 6 imports these.
TEMPERATURE = SensorChannel(
    name="temperature", unit="C", phys_min=150.0, phys_max=400.0, noise_std=1.5
)
PRESSURE = SensorChannel(
    name="pressure", unit="bar", phys_min=0.0, phys_max=50.0, noise_std=0.2
)


def characterize(ch: SensorChannel, rng: np.random.Generator, n: int = 20000) -> dict:
    """Empirically characterise a channel: RMS error and effective number of bits."""
    mid = 0.5 * (ch.phys_min + ch.phys_max)
    truth = np.full(n, mid)
    reading = ch.read(truth, rng)
    rms = float(np.sqrt(np.mean((reading - truth) ** 2)))
    full_scale = ch.phys_max - ch.phys_min
    # Effective number of bits from the achieved RMS error vs full scale.
    enob = float(np.log2(full_scale / (rms * np.sqrt(12)))) if rms > 0 else float(ch.adc_bits)
    return {
        "channel": ch.name,
        "quantization_step": ch.quantization_step,
        "sensor_noise_std": ch.noise_std,
        "rms_error": rms,
        "enob": enob,
    }


def demo(*, show: bool) -> Path:
    """Corrupt a stable reactor temperature trajectory and plot clean vs. noisy."""
    from sim import params as P
    from sim.reactor import simulate

    rng = np.random.default_rng(42)
    res = simulate(P.NOMINAL_OP, P.NOMINAL_CONFIG, t_end=120.0, n_points=240)
    t = res.t
    clean_c = res.temperature - 273.15
    noisy_c = TEMPERATURE.read(clean_c, rng)

    import matplotlib.pyplot as plt

    fig, (ax_t, ax_h) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax_t.plot(t, clean_c, "k", lw=1.5, label="clean physics")
    ax_t.plot(t, noisy_c, "tab:orange", lw=0.8, alpha=0.8, label="sensor + 12-bit ADC")
    ax_t.set(title="Reactor temperature telemetry", xlabel="time (s)", ylabel="T (C)")
    ax_t.legend(fontsize=9)

    err = noisy_c - clean_c
    ax_h.hist(err, bins=40, color="tab:orange", alpha=0.8)
    ax_h.set(title=f"Reading error (RMS={np.sqrt(np.mean(err**2)):.2f} C)",
             xlabel="noisy - clean (C)", ylabel="count")

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "sensor_noise.png"
    fig.savefig(out, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(0)
    for ch in (TEMPERATURE, PRESSURE):
        c = characterize(ch, rng)
        print(
            f"{c['channel']:>11}: quant step = {c['quantization_step']:.4f} {ch.unit}, "
            f"sensor sigma = {c['sensor_noise_std']} {ch.unit}, "
            f"RMS = {c['rms_error']:.3f} {ch.unit}, ENOB = {c['enob']:.1f} bits"
        )
    out = demo(show=not args.no_show)
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
