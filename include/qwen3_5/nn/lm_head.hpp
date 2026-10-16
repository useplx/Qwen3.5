#pragma once

#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Qwen35::Nn {
    class LMHead : public Mln::Sdk::Module {
        public:
            LMHead();
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &hidden);
    };
}
