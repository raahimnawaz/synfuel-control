"""Physics-informed loss: data term + steady-state conservation residuals.

The "physics-informed" part penalises the network's predicted state for violating the
conservation laws that must hold at any CSTR steady state, expressed as *relative*
(dimensionless, bounded) residuals so they are well-conditioned:

  * **carbon balance**  — carbon fed (as CO2) equals carbon out (CO2 + CO + HC);
  * **hydrogen balance** — hydrogen fed (as H2) equals hydrogen out (H2 + H2O + HC);
  * **energy balance**  — convective enthalpy change + heat of reaction = jacket cooling.

Why relative residuals rather than the raw dy/dt: at steady state large production and
consumption terms cancel, so the bare right-hand side is numerically stiff — a near
perfect fit can still show an enormous residual, and minimising it fights the data.
Normalising each balance by the magnitude of its own terms keeps every residual in
roughly [-1, 1] and makes the physics term a clean regulariser that the true steady
states already satisfy (so it never fights the data) while constraining the surrogate
to stay physically consistent where training data is sparse.
"""

from __future__ import annotations

import torch

from sim import params as P

# Fixed reactor constants (ReactorConfig defaults); cooling_ua comes from the inputs.
_SV = P.NOMINAL_CONFIG.space_velocity
_RHO_CP = P.NOMINAL_CONFIG.rho_cp
_FEED_T = P.NOMINAL_CONFIG.feed_temp
_TAU = 1.0 / _SV  # characteristic residence time, used to non-dimensionalise residuals


def _feed_concentrations(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ratio, pressure = inputs[:, 0], inputs[:, 1]
    c_total = (pressure * 1.0e5) / (P.R_GAS * _FEED_T)
    x_h2 = ratio / (1.0 + ratio)
    x_co2 = 1.0 / (1.0 + ratio)
    return x_co2 * c_total, x_h2 * c_total


def reactor_rhs(state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Differentiable dy/dt for the CSTR — mirrors sim.reactor.derivatives."""
    c_co2, c_h2, c_co, c_h2o, _c_hc, temp = (state[:, i] for i in range(6))
    coolant_temp, catalyst, cooling_ua = inputs[:, 2], inputs[:, 3], inputs[:, 4]
    c_co2_in, c_h2_in = _feed_concentrations(inputs)

    k1f = P.A_RWGS * torch.exp(-P.EA_RWGS / (P.R_GAS * temp))
    keq = 1.0e-5 * torch.exp(-P.DH_RWGS / P.R_GAS * (1.0 / temp - 1.0 / 298.15))
    k1r = k1f / keq
    r1 = k1f * c_co2 * c_h2 - k1r * c_co * c_h2o

    k2 = catalyst * P.A_FT * torch.exp(-P.EA_FT / (P.R_GAS * temp))
    r2 = k2 * c_co * c_h2

    d_co2 = _SV * (c_co2_in - c_co2) - r1
    d_h2 = _SV * (c_h2_in - c_h2) - r1 - 2.0 * r2
    d_co = _SV * (-c_co) + r1 - r2
    d_h2o = _SV * (-c_h2o) + r1 + r2
    d_hc = _SV * (-_c_hc) + r2

    heat_release = (-P.DH_RWGS) * r1 + (-P.DH_FT) * r2
    cooling = cooling_ua * (temp - coolant_temp)
    d_temp = _SV * (_FEED_T - temp) + (heat_release - cooling) / _RHO_CP

    return torch.stack([d_co2, d_h2, d_co, d_h2o, d_hc, d_temp], dim=1)


def conservation_residuals(state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Relative carbon, hydrogen, and energy steady-state residuals, shape (B, 3)."""
    eps = 1.0e-6
    c_co2, c_h2, c_co, c_h2o, c_hc, temp = (state[:, i] for i in range(6))
    coolant_temp, catalyst, cooling_ua = inputs[:, 2], inputs[:, 3], inputs[:, 4]
    c_co2_in, c_h2_in = _feed_concentrations(inputs)

    # Carbon: CO2 in == CO2 + CO + HC out (one C each).
    carbon = (c_co2_in - (c_co2 + c_co + c_hc)) / (c_co2_in + eps)
    # Hydrogen: H2 in == H2 + H2O + HC out (two H each).
    hydrogen = (c_h2_in - (c_h2 + c_h2o + c_hc)) / (c_h2_in + eps)

    # Energy: sv*rho_cp*(feed_T - T) + heat_release - cooling == 0, normalised by term size.
    k1f = P.A_RWGS * torch.exp(-P.EA_RWGS / (P.R_GAS * temp))
    keq = 1.0e-5 * torch.exp(-P.DH_RWGS / P.R_GAS * (1.0 / temp - 1.0 / 298.15))
    r1 = k1f * c_co2 * c_h2 - (k1f / keq) * c_co * c_h2o
    r2 = catalyst * P.A_FT * torch.exp(-P.EA_FT / (P.R_GAS * temp)) * c_co * c_h2
    heat_release = (-P.DH_RWGS) * r1 + (-P.DH_FT) * r2
    cooling = cooling_ua * (temp - coolant_temp)
    convective = _SV * _RHO_CP * (_FEED_T - temp)
    energy = (convective + heat_release - cooling) / (
        convective.abs() + heat_release.abs() + cooling.abs() + eps
    )
    return torch.stack([carbon, hydrogen, energy], dim=1)


def combined_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_true: torch.Tensor,
    *,
    w_phys: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, data, physics) losses.

    Data loss is MSE in normalised state space (so all six variables count equally);
    the physics term is the mean-square conservation residual at the predicted state.
    """
    pred = model(x)
    data = torch.mean(((pred - y_true) / model.out_std) ** 2)
    phys = torch.mean(conservation_residuals(pred, x) ** 2)
    return data + w_phys * phys, data, phys
