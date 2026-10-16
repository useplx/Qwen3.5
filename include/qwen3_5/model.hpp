#pragma once

#include "qwen3_5/nn/decoder.hpp"
#include "qwen3_5/vision.hpp"

#include <mln/sdk/orchestration.hpp>

namespace Qwen35 {
    class Model : public Mln::Sdk::Orchestration {
        public:
            explicit Model(Config config = {}) : config_(config) {
                config_.validate();
                name = "qwen35";
                modules.insert<Nn::Decoder>("model", config);
            }

            const Config &config() const { return config_; }

        protected:
            Config config_;

            Mln::Buffer tokenize(Mln::Buffer text, const Mln::Buffer &tokenizer_asset);

            Mln::Buffer tokenize_chat(Mln::Buffer system_and_user,
                                      const Mln::Buffer &tokenizer_asset);

            Mln::Buffer tokenize_visual(std::span<const Content> content,
                                        std::span<const Vision::Grid> grids,
                                        const Mln::Buffer &tokenizer_asset);

            Mln::Buffer detokenize(Mln::Buffer tokens, const Mln::Buffer &tokenizer_asset);

            Mln::Buffer generate(Mln::Runtime::Session &session, Mln::Buffer prompt_tokens,
                                 const Mln::Buffer &tokenizer_asset);
            Mln::Buffer generate(Mln::Runtime::Session &session, Mln::Buffer prompt_tokens,
                                 const Mln::Buffer &tokenizer_asset,
                                 const Mln::Buffer &visual_embeddings,
                                 std::span<const Vision::Grid> grids);
    };
}
