"""Physical constants and reactor configuration for the Fischer-Tropsch synfuel reactor.

Every *physical* constant below is taken from the open literature and carries an inline
citation. The only tuned quantities are the two rate pre-exponential factors
(``A_FT``, ``A_RWGS``): absolute pre-exponentials are catalyst-, loading-, and
support-specific and are routinely fitted per reactor, so we fit them to give physically
reasonable conversion at the nominal operating point. The *temperature dependence*
(activation energies), the *reaction enthalpies*, and the *chain-growth probability* —
the things that actually govern the dynamics and the runaway behaviour — are all from
literature. See sim/CHEMISTRY.md for the full discussion and sources.

References
----------
[Dry2002]   M. E. Dry, "The Fischer-Tropsch process: 1950-2000",
            Catalysis Today 71 (2002) 227-241.
[vdLaan99]  G. P. van der Laan & A. A. C. M. Beenackers, "Kinetics and Selectivity of
            the Fischer-Tropsch Synthesis: A Literature Review",
            Catalysis Reviews 41 (1999) 255-318.
[Yates91]   I. C. Yates & C. N. Satterfield, "Intrinsic kinetics of the Fischer-Tropsch
            synthesis on a cobalt catalyst", Energy & Fuels 5 (1991) 168-173.
[NIST]      NIST Chemistry WebBook, standard enthalpies of formation (RWGS at 298 K).
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Universal constants -------------------------------------------------------------
R_GAS = 8.314462618  # J / (mol * K), CODATA

# --- Reaction thermochemistry (literature) -------------------------------------------
# Fischer-Tropsch is strongly exothermic per CO converted to a -CH2- chain unit; this
# exothermicity is the physical origin of thermal runaway. [Dry2002] gives ~ -165 kJ/mol.
DH_FT = -165.0e3  # J / mol CO converted  [Dry2002]

# Reverse water-gas-shift (CO2 + H2 -> CO + H2O) is mildly endothermic. Standard
# enthalpy at 298 K from formation enthalpies. [NIST]
DH_RWGS = +41.2e3  # J / mol  [NIST]

# --- Apparent activation energies (literature) ---------------------------------------
# Cobalt FT apparent activation energy clusters around 100 kJ/mol across studies.
EA_FT = 100.0e3  # J / mol  [Yates91, vdLaan99]
# RWGS apparent activation energy over typical Fe/Cu catalysts ~ 70 kJ/mol.
EA_RWGS = 70.0e3  # J / mol  [vdLaan99]

# --- Anderson-Schulz-Flory chain growth ----------------------------------------------
# Chain-growth probability for low-temperature cobalt FT; high alpha favours the heavy
# (C5+, "liquid fuel") fraction. [Dry2002, vdLaan99]
ASF_ALPHA = 0.90

# --- Tuned rate pre-exponentials (catalyst-specific; see module docstring) ------------
# Units: m^3 / (mol * s) for the second-order rate laws used in reactor.py.
# Fitted so the nominal operating point gives ~33% CO2 conversion at ~227 C, a
# physically reasonable low-temperature cobalt FT operating window.
A_FT = 3.0e8
A_RWGS = 2.5e5


@dataclass(frozen=True)
class ReactorConfig:
    """Fixed geometry / transport properties of the (lumped CSTR) reactor.

    The reactor is modelled as a continuous stirred-tank reactor: a single
    well-mixed gas volume packed with catalyst, fed continuously and cooled through
    a jacket. ``rho_cp`` is the effective volumetric heat capacity of the catalyst
    bed (solid-dominated), which sets the thermal time constant.
    """

    space_velocity: float = 0.2   # 1/s   (q/V; residence time = 5 s)
    rho_cp: float = 1.0e5         # J/(m^3*K)  effective bed volumetric heat capacity
    ua_per_volume: float = 6.0e4  # W/(m^3*K)  jacket heat-transfer coefficient * area / V
    feed_temp: float = 470.0      # K     (~197 C) feed gas temperature

    # Safety / quality limit. Above this the cobalt catalyst sinters and CH4
    # selectivity spikes; we treat it as the thermal-runaway threshold.
    runaway_temp: float = 575.0   # K  (~302 C)


@dataclass(frozen=True)
class OperatingPoint:
    """Manipulated variables — what the (future) controller actuates."""

    h2_co2_ratio: float = 3.0     # feed H2 : CO2 molar ratio
    pressure_bar: float = 25.0    # total reactor pressure
    coolant_temp: float = 490.0   # K  jacket coolant temperature
    catalyst: float = 1.0         # catalyst-loading multiplier (1.0 = nominal)


# Convenience singletons for the nominal design point.
NOMINAL_CONFIG = ReactorConfig()
NOMINAL_OP = OperatingPoint()
