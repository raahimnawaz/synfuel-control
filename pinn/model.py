"""The surrogate network: 5 operating inputs -> 6-dim reactor steady state.

A deliberately small MLP (so Phase 5 can hand-roll the forward pass in C++). Input
scaling and output de-normalisation are baked into the module as buffers, so the
exported ONNX graph is fully self-contained: it takes raw physical inputs and returns
the physical steady state ``[C_CO2, C_H2, C_CO, C_H2O, C_HC, T]`` (mol/m^3 x5, then K).

Predicting the full state (rather than just yield) is what lets the physics-informed
loss enforce the governing CSTR equations on the output — see losses.py.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

INPUT_DIM = 5   # h2_co2_ratio, pressure_bar, coolant_temp, catalyst, cooling_ua
STATE_DIM = 6   # C_CO2, C_H2, C_CO, C_H2O, C_HC, T


class ReactorSurrogate(nn.Module):
    """Small tanh MLP mapping operating inputs to the reactor steady state."""

    def __init__(
        self,
        in_lows: np.ndarray,
        in_highs: np.ndarray,
        out_mean: np.ndarray,
        out_std: np.ndarray,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        # Normalisation constants stored as buffers (exported into the ONNX graph).
        self.register_buffer("in_lows", torch.as_tensor(in_lows, dtype=torch.float32))
        self.register_buffer("in_highs", torch.as_tensor(in_highs, dtype=torch.float32))
        self.register_buffer("out_mean", torch.as_tensor(out_mean, dtype=torch.float32))
        self.register_buffer("out_std", torch.as_tensor(out_std, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, STATE_DIM),
        )

    def _scale_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw inputs to [-1, 1] using the design-space bounds."""
        return 2.0 * (x - self.in_lows) / (self.in_highs - self.in_lows) - 1.0

    def forward_normalised(self, x: torch.Tensor) -> torch.Tensor:
        """Network output in normalised (z-scored) state space."""
        return self.net(self._scale_inputs(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted steady state in physical units."""
        return self.forward_normalised(x) * self.out_std + self.out_mean
