#include "mlp.h"

#include <cmath>

#include "weights.h"

namespace synfuel {

void predict(const float in[5], float out[6]) {
    // 1. Scale inputs to [-1, 1] using the design-space bounds (matches the model).
    float z[IN_DIM];
    for (int i = 0; i < IN_DIM; ++i) {
        z[i] = 2.0f * (in[i] - IN_LOWS[i]) / (IN_HIGHS[i] - IN_LOWS[i]) - 1.0f;
    }

    // 2. Hidden layer 1: tanh(W0 z + b0).
    float h1[H1];
    for (int i = 0; i < H1; ++i) {
        float acc = B0[i];
        for (int j = 0; j < IN_DIM; ++j) acc += W0[i][j] * z[j];
        h1[i] = std::tanh(acc);
    }

    // 3. Hidden layer 2: tanh(W1 h1 + b1).
    float h2[H2];
    for (int i = 0; i < H2; ++i) {
        float acc = B1[i];
        for (int j = 0; j < H1; ++j) acc += W1[i][j] * h1[j];
        h2[i] = std::tanh(acc);
    }

    // 4. Output layer + de-normalisation back to physical units.
    for (int i = 0; i < OUT_DIM; ++i) {
        float acc = B2[i];
        for (int j = 0; j < H2; ++j) acc += W2[i][j] * h2[j];
        out[i] = acc * OUT_STD[i] + OUT_MEAN[i];
    }
}

}  // namespace synfuel
