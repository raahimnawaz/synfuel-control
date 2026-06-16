"""First-principles dynamic model of a Fischer-Tropsch synfuel reactor (the "plant").

A lumped continuous-stirred-tank reactor (CSTR) converting a CO2 + H2 feed into a
lumped hydrocarbon product via two coupled reactions:

    (1) reverse water-gas-shift   CO2 + H2  <->  CO + H2O      (mildly endothermic)
    (2) Fischer-Tropsch growth    CO + 2 H2  ->  (-CH2-) + H2O (strongly exothermic)

The state is the species concentrations plus the bed temperature::

    y = [C_CO2, C_H2, C_CO, C_H2O, C_HC, T]    (mol/m^3 x5, then K)

``C_HC`` counts -CH2- monomer units (i.e. carbon atoms locked into product); the
Anderson-Schulz-Flory distribution then maps that to a C5+ (liquid-fuel) fraction.

The Arrhenius temperature dependence of reaction (2), combined with its large
exothermicity, is what produces thermal runaway: hotter -> faster -> more heat.
The whole point of the downstream controller is to keep T below the runaway limit
while maximising C5+ yield.

All physical constants live in params.py with literature citations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from . import params as P

# Indices into the state vector.
I_CO2, I_H2, I_CO, I_H2O, I_HC, I_T = range(6)
SPECIES = ("CO2", "H2", "CO", "H2O", "HC")


def arrhenius(pre_exp: float, ea: float, temp: float | np.ndarray) -> float | np.ndarray:
    """Arrhenius rate constant k = A * exp(-Ea / (R*T))."""
    return pre_exp * np.exp(-ea / (P.R_GAS * temp))


def rwgs_equilibrium_constant(temp: float | np.ndarray) -> float | np.ndarray:
    """Van't Hoff estimate of the RWGS equilibrium constant, anchored at 298 K.

    Keq(T) = Keq(298) * exp(-dH/R * (1/T - 1/298)). RWGS is endothermic so Keq rises
    with temperature, which is why higher T favours CO formation (and thus FT feed).
    """
    keq_298 = 1.0e-5  # RWGS strongly disfavoured at room temperature
    return keq_298 * np.exp(-P.DH_RWGS / P.R_GAS * (1.0 / temp - 1.0 / 298.15))


def feed_concentrations(op: P.OperatingPoint, cfg: P.ReactorConfig) -> tuple[float, float]:
    """Inlet CO2 and H2 concentrations (mol/m^3) from pressure and feed ratio.

    Ideal gas: total concentration = P / (R * T_feed). The feed is H2 + CO2 only,
    split by the molar ratio.
    """
    c_total = (op.pressure_bar * 1.0e5) / (P.R_GAS * cfg.feed_temp)
    x_h2 = op.h2_co2_ratio / (1.0 + op.h2_co2_ratio)
    x_co2 = 1.0 / (1.0 + op.h2_co2_ratio)
    return x_co2 * c_total, x_h2 * c_total


def reaction_rates(
    y: np.ndarray, op: P.OperatingPoint
) -> tuple[float, float]:
    """Volumetric rates (mol/m^3/s) of (1) RWGS and (2) Fischer-Tropsch growth."""
    c_co2, c_h2, c_co, c_h2o, _c_hc, temp = y

    k1f = arrhenius(P.A_RWGS, P.EA_RWGS, temp)
    k1r = k1f / rwgs_equilibrium_constant(temp)
    r_rwgs = k1f * c_co2 * c_h2 - k1r * c_co * c_h2o

    k2 = op.catalyst * arrhenius(P.A_FT, P.EA_FT, temp)
    r_ft = k2 * c_co * c_h2

    return r_rwgs, r_ft


def derivatives(
    t: float,  # noqa: ARG001 - solve_ivp signature
    y: np.ndarray,
    op: P.OperatingPoint,
    cfg: P.ReactorConfig,
) -> np.ndarray:
    """Right-hand side dy/dt for the CSTR (species balances + energy balance)."""
    sv = cfg.space_velocity
    c_co2_in, c_h2_in = feed_concentrations(op, cfg)
    r1, r2 = reaction_rates(y, op)
    temp = y[I_T]

    dy = np.empty(6)
    # Species: inflow/outflow + net generation by the two reactions.
    dy[I_CO2] = sv * (c_co2_in - y[I_CO2]) - r1
    dy[I_H2] = sv * (c_h2_in - y[I_H2]) - r1 - 2.0 * r2
    dy[I_CO] = sv * (0.0 - y[I_CO]) + r1 - r2
    dy[I_H2O] = sv * (0.0 - y[I_H2O]) + r1 + r2
    dy[I_HC] = sv * (0.0 - y[I_HC]) + r2

    # Energy: convective in/out + heat of reaction - jacket cooling.
    heat_release = (-P.DH_RWGS) * r1 + (-P.DH_FT) * r2  # W/m^3
    cooling = cfg.ua_per_volume * (temp - op.coolant_temp)  # W/m^3
    dy[I_T] = (
        sv * (cfg.feed_temp - temp)
        + (heat_release - cooling) / cfg.rho_cp
    )
    return dy


# --- Anderson-Schulz-Flory product distribution --------------------------------------
def asf_weight_fraction(n: int | np.ndarray, alpha: float = P.ASF_ALPHA) -> float | np.ndarray:
    """ASF mass fraction of carbon-number n: w_n = n (1-alpha)^2 alpha^(n-1)."""
    return n * (1.0 - alpha) ** 2 * alpha ** (n - 1)


def c5plus_fraction(alpha: float = P.ASF_ALPHA) -> float:
    """Mass fraction of product in the C5+ (liquid-fuel) range.

    Computed as 1 minus the C1..C4 fractions, which have a closed ASF form.
    """
    light = sum(asf_weight_fraction(n, alpha) for n in (1, 2, 3, 4))
    return 1.0 - light


@dataclass
class ReactorResult:
    """Outcome of a simulation run."""

    t: np.ndarray            # time grid (s)
    y: np.ndarray            # state trajectory, shape (6, len(t))
    op: P.OperatingPoint
    cfg: P.ReactorConfig

    @property
    def temperature(self) -> np.ndarray:
        return self.y[I_T]

    @property
    def runaway(self) -> bool:
        """Did the bed exceed the safety/runaway temperature at any point?"""
        return bool(np.any(self.temperature >= self.cfg.runaway_temp))

    def co2_conversion(self) -> np.ndarray:
        """Fraction of fed carbon (CO2) that left as CO or hydrocarbon."""
        c_co2_in, _ = feed_concentrations(self.op, self.cfg)
        return (c_co2_in - self.y[I_CO2]) / c_co2_in

    def c5plus_yield(self) -> np.ndarray:
        """C5+ liquid-fuel yield relative to carbon fed (0..1)."""
        c_co2_in, _ = feed_concentrations(self.op, self.cfg)
        return (self.y[I_HC] / c_co2_in) * c5plus_fraction(P.ASF_ALPHA)


def simulate(
    op: P.OperatingPoint = P.NOMINAL_OP,
    cfg: P.ReactorConfig = P.NOMINAL_CONFIG,
    *,
    t_end: float = 200.0,
    n_points: int = 800,
    y0: np.ndarray | None = None,
) -> ReactorResult:
    """Integrate the reactor ODEs from a cold start to ``t_end`` seconds.

    Uses a stiff solver (BDF) because the Arrhenius source term makes the energy
    equation stiff, especially near runaway.
    """
    if y0 is None:
        # Start filled with feed gas (CO2 + H2) at feed temperature, no products yet.
        c_co2_in, c_h2_in = feed_concentrations(op, cfg)
        y0 = np.array([c_co2_in, c_h2_in, 0.0, 0.0, 0.0, cfg.feed_temp])

    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        derivatives,
        (0.0, t_end),
        y0,
        args=(op, cfg),
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-8,
        max_step=t_end / 50.0,
    )
    if not sol.success:
        raise RuntimeError(f"reactor integration failed: {sol.message}")
    return ReactorResult(t=sol.t, y=sol.y, op=op, cfg=cfg)
