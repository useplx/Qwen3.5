#pragma once

#include "qwen3_vl/model.hpp"

namespace Qwen3VL {
    class Generator final : public Model {
        public:
            using Model::Model;

            Mln::Buffer tokenize_chat(Mln::Buffer system_and_user,
                                      const Mln::Buffer &tokenizer_asset) {
                return Model::tokenize_chat(std::move(system_and_user), tokenizer_asset);
            }

            Mln::Buffer generate(Mln::Runtime::Session &session, Mln::Buffer prompt_tokens,
                                 const Mln::Buffer &tokenizer_asset) {
                return Model::generate(session, std::move(prompt_tokens), tokenizer_asset);
            }

            const char *tokenizer_asset_name() const { return "tokenizer.json"; }

            Mln::Buffer forward(Mln::Runtime::Session &, Mln::Buffer input) override {
                return input;
            }
    };
}
