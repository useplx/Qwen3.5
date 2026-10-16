#pragma once

#include <cstdint>
#include <vector>

namespace Qwen35 {
    void rope_cos_sin(int64_t t_pos, int64_t h_pos, int64_t w_pos, float *cos_out, float *sin_out);

    void causal_mask(int64_t pos, int64_t length, float *mask_out);
}
