#pragma once

#include "qwen3_5/config.hpp"

namespace Qwen35::Paths {
    inline constexpr const char *MODEL = "model";
    inline constexpr const char *EMBED_TOKENS = "model.language_model.embed_tokens";
    inline constexpr const char *LAYERS = "model.language_model.layers.*";
    inline constexpr const char *LAYER_INPUT_NORM = "model.language_model.layers.*.input_layernorm";
    inline constexpr const char *LAYER_ATTENTION = "model.language_model.layers.*.self_attn";
    inline constexpr const char *LAYER_POST_ATTENTION_NORM =
        "model.language_model.layers.*.post_attention_layernorm";
    inline constexpr const char *LAYER_MLP = "model.language_model.layers.*.mlp";
    inline constexpr const char *HEAD = "model.language_model.head";
    inline constexpr const char *VISION = "model.visual";
    inline constexpr const char *VISION_BLOCKS = "model.visual.blocks.*";
    inline constexpr const char *VISION_MERGER = "model.visual.merger";
}
