// Hand-rolled inference for the reactor surrogate — zero runtime dependencies.
//
// Targets the Jetson Orin Nano edge architecture; the network is small enough that a
// from-scratch forward pass is both trivial and far faster than loading a runtime.
#pragma once

namespace synfuel {

// Predict the reactor steady state from the operating inputs.
//   in  = {h2_co2_ratio, pressure_bar, coolant_temp[K], catalyst, cooling_ua}
//   out = {C_CO2, C_H2, C_CO, C_H2O, C_HC [mol/m^3], T [K]}
void predict(const float in[5], float out[6]);

}  // namespace synfuel
