#pragma once

#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

namespace Qwen35::Nn {
    class Embedding : public Mln::Sdk::Module {
        public:
            Embedding();
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &token_id);
    };
}
