#pragma once

#include "qwen3_5/errors.hpp"

#include <cstdint>

#if !defined(QWEN35_MAX_SEQ_LEN) || !defined(QWEN35_MAX_NEW_TOKENS) ||                           \
    !defined(QWEN35_PREFILL_CHUNK) || !defined(QWEN35_PREFILL_MIN_CHUNK) ||                      \
    !defined(QWEN35_VISION_MAX_PATCHES) || !defined(QWEN35_DO_SAMPLE) ||                         \
    !defined(QWEN35_TOP_K) || !defined(QWEN35_TOP_P) || !defined(QWEN35_TEMPERATURE) ||         \
    !defined(QWEN35_REPETITION_PENALTY) || !defined(QWEN35_PRESENCE_PENALTY)
#error "Qwen3.5 configuration must be supplied by synps.toml [package.variables]"
#endif

namespace Qwen35 {
    struct Config {
            int64_t max_seq_len = QWEN35_MAX_SEQ_LEN;

            int64_t max_new_tokens = QWEN35_MAX_NEW_TOKENS;

            int64_t prefill_chunk = QWEN35_PREFILL_CHUNK;
            int64_t prefill_min_chunk = QWEN35_PREFILL_MIN_CHUNK;

            int64_t vision_max_patches = QWEN35_VISION_MAX_PATCHES;

            bool do_sample = QWEN35_DO_SAMPLE;
            int64_t top_k = QWEN35_TOP_K;
            double top_p = QWEN35_TOP_P;
            double temperature = QWEN35_TEMPERATURE;
            double repetition_penalty = QWEN35_REPETITION_PENALTY;
            double presence_penalty = QWEN35_PRESENCE_PENALTY;

            void validate() const {
                if (max_seq_len < 2)
                    Mln::raise(Errors::Config::Context::Invalid, max_seq_len);
                if (max_new_tokens < 1)
                    Mln::raise(Errors::Config::Generated::Token::Count::Invalid, max_new_tokens);
                if (prefill_chunk < 1 || (prefill_chunk & (prefill_chunk - 1)) != 0)
                    Mln::raise(Errors::Config::Prefill::Chunk::Invalid, prefill_chunk);
                if (prefill_min_chunk < 2 || prefill_min_chunk > prefill_chunk ||
                    (prefill_min_chunk & (prefill_min_chunk - 1)) != 0)
                    Mln::raise(Errors::Config::Prefill::Chunk::Invalid, prefill_min_chunk);
                if (vision_max_patches < 256 ||
                    (vision_max_patches & (vision_max_patches - 1)) != 0)
                    Mln::raise(Errors::Config::Vision::Patches::Invalid, vision_max_patches);
            }
    };
}

namespace Qwen35::Arch {
    inline constexpr int64_t HIDDEN_SIZE = 2048;
    inline constexpr int64_t NUM_LAYERS = 24;
    inline constexpr int64_t NUM_HEADS = 8;
    inline constexpr int64_t NUM_KV_HEADS = 2;
    inline constexpr int64_t HEAD_DIM = 256;
    inline constexpr int64_t INTERMEDIATE_SIZE = 6144;
    inline constexpr int64_t VOCAB_SIZE = 248320;
    inline constexpr double RMS_NORM_EPS = 1e-6;
    inline constexpr double ROPE_THETA = 10000000.0;
    inline constexpr int64_t ROTARY_DIM = 64;
    inline constexpr int64_t FULL_ATTENTION_INTERVAL = 4;

    inline constexpr int64_t LINEAR_NUM_HEADS = 16;
    inline constexpr int64_t LINEAR_HEAD_DIM = 128;
    inline constexpr int64_t LINEAR_KEY_SIZE = LINEAR_NUM_HEADS * LINEAR_HEAD_DIM;
    inline constexpr int64_t LINEAR_VALUE_SIZE = LINEAR_NUM_HEADS * LINEAR_HEAD_DIM;
    inline constexpr int64_t LINEAR_CONV_DIM = 2 * LINEAR_KEY_SIZE + LINEAR_VALUE_SIZE;
    inline constexpr int64_t LINEAR_CONV_KERNEL = 4;
    inline constexpr int64_t MROPE_SECTION_T = 11;
    inline constexpr int64_t MROPE_SECTION_H = 11;
    inline constexpr int64_t MROPE_SECTION_W = 10;

    inline constexpr int64_t IMAGE_TOKEN_ID = 248056;
    inline constexpr int64_t VIDEO_TOKEN_ID = 248057;
    inline constexpr int64_t VISION_START_TOKEN_ID = 248053;
    inline constexpr int64_t VISION_END_TOKEN_ID = 248054;
    inline constexpr int64_t VISION_PAD_TOKEN_ID = 248055;
    inline constexpr int64_t IM_START_TOKEN_ID = 248045;
    inline constexpr int64_t IM_END_TOKEN_ID = 248046;
    inline constexpr int64_t BOS_TOKEN_ID = 248044;
    inline constexpr int64_t EOS_TOKEN_ID = 248044;
    inline constexpr int64_t END_OF_TEXT_TOKEN_ID = 248044;

    inline constexpr int64_t VISION_HIDDEN_SIZE = 1024;
    inline constexpr int64_t VISION_DEPTH = 24;
    inline constexpr int64_t VISION_NUM_HEADS = 16;
    inline constexpr int64_t VISION_INTERMEDIATE_SIZE = 4096;
    inline constexpr int64_t VISION_PATCH_SIZE = 16;
    inline constexpr int64_t VISION_TEMPORAL_PATCH_SIZE = 2;
    inline constexpr int64_t VISION_SPATIAL_MERGE_SIZE = 2;
    inline constexpr int64_t VISION_OUT_HIDDEN_SIZE = 2048;
    inline constexpr int64_t VISION_IN_CHANNELS = 3;
}
