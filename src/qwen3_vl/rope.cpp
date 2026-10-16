#include "qwen3_vl/rope.hpp"
#include "qwen3_vl/config.hpp"

#include <cmath>

using namespace Qwen3VL::Arch;

namespace Qwen3VL {
    void rope_cos_sin(int64_t t_pos, int64_t h_pos, int64_t w_pos, float *cos_out, float *sin_out) {
        constexpr int64_t half = ROTARY_DIM / 2;

        for (int64_t i = 0; i < half; i++) {
            const double inv_freq = 1.0 / std::pow(ROPE_THETA, static_cast<double>(2 * i) /
                                                                   static_cast<double>(ROTARY_DIM));
            int64_t pos = t_pos;
            const int64_t phase = i % 3;
            if (phase == 1 && i < 3 * MROPE_SECTION_H)
                pos = h_pos;
            else if (phase == 2 && i < 3 * MROPE_SECTION_W)
                pos = w_pos;
            const double angle = static_cast<double>(pos) * inv_freq;
            const float c = static_cast<float>(std::cos(angle));
            const float s = static_cast<float>(std::sin(angle));
            cos_out[i] = c;
            cos_out[i + half] = c;
            sin_out[i] = s;
            sin_out[i + half] = s;
        }
    }

    void causal_mask(int64_t pos, int64_t length, float *mask_out) {
        for (int64_t i = 0; i < length; i++)
            mask_out[i] = i <= pos ? 0.0f : -1.0e9f;
    }
}
