#pragma once

#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

#include <span>

namespace Qwen35::Nn {
    class Attention : public Mln::Sdk::Module {
        public:
            struct Output {
                    Mln::Sdk::Tensor hidden;
                    Mln::Sdk::Tensor k_cache;
                    Mln::Sdk::Tensor v_cache;

                    void apply_hook(Mln::Sdk::Module &module,
                                    std::span<const Mln::Sdk::Tensor> inputs) {
                        hidden = module.apply(std::move(hidden), inputs, "hidden");
                        k_cache = module.apply(std::move(k_cache), inputs, "k_cache");
                        v_cache = module.apply(std::move(v_cache), inputs, "v_cache");
                    }
            };

            Attention();

            Output forward(const Mln::Sdk::Tensor &hidden, const Mln::Sdk::Tensor &cos,
                           const Mln::Sdk::Tensor &sin, const Mln::Sdk::Tensor &k_cache_in,
                           const Mln::Sdk::Tensor &v_cache_in, const Mln::Sdk::Tensor &pos,
                           const Mln::Sdk::Tensor &mask);
    };
}
