#pragma once

#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

#include <cstdint>

namespace Qwen3VL::Nn {
    class Linear : public Mln::Sdk::Module {
        public:
            Linear(int64_t in_features, int64_t out_features);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            int64_t in_features_;
            int64_t out_features_;
    };
}
