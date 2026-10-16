#pragma once

#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Qwen35::Nn {
    class GatedDeltaNet : public Mln::Sdk::Module {
        public:
            struct Output {
                    Mln::Sdk::Tensor hidden;
                    Mln::Sdk::Tensor convolution;
                    Mln::Sdk::Tensor recurrent;
                    void apply_hook(Mln::Sdk::Module &module,
                                    std::span<const Mln::Sdk::Tensor> inputs);
            };

            GatedDeltaNet();

            Output forward(const Mln::Sdk::Tensor &hidden, const Mln::Sdk::Tensor &convolution,
                           const Mln::Sdk::Tensor &recurrent);
    };
}
