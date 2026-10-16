#pragma once

#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Qwen3VL::Nn {
    class MLP : public Mln::Sdk::Module {
        public:
            MLP();
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);
    };
}
