#pragma once

#include "qwen3_5/nn/decoder.hpp"

#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

#include <cstdint>
#include <span>

namespace Qwen35::Nn {
    class Decoder::Layer : public Mln::Sdk::Module {
        public:
            struct Output {
                    Mln::Sdk::Tensor hidden;
                    Mln::Sdk::Tensor state_a;
                    Mln::Sdk::Tensor state_b;
                    void apply_hook(Mln::Sdk::Module &module,
                                    std::span<const Mln::Sdk::Tensor> inputs) {
                        hidden = module.apply(std::move(hidden), inputs, "hidden");
                        state_a = module.apply(std::move(state_a), inputs, "state_a");
                        state_b = module.apply(std::move(state_b), inputs, "state_b");
                    }
            };

            explicit Layer(int64_t layer_idx);

            Output forward(const Mln::Sdk::Tensor &hidden, const Mln::Sdk::Tensor &cos,
                           const Mln::Sdk::Tensor &sin, const Mln::Sdk::Tensor &state_a,
                           const Mln::Sdk::Tensor &state_b, const Mln::Sdk::Tensor &position,
                           const Mln::Sdk::Tensor &mask);

        private:
            bool full_attention_;
    };
}
