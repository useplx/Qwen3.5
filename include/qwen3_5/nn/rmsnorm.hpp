#pragma once

#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

#include <cstdint>

namespace Qwen35::Nn {
    Mln::Sdk::Tensor rms_norm(Mln::Sdk::Operations &ops, const Mln::Sdk::Tensor &x,
                              const Mln::Sdk::Tensor &weight, int64_t dim, double eps,
                              bool zero_centered = false);
    Mln::Sdk::Tensor rms_norm(Mln::Sdk::Operations &ops, const Mln::Sdk::Tensor &x,
                              const Mln::Sdk::Tensor &weight, int64_t dim, double eps,
                              Mln::DType computation_dtype, bool zero_centered = false);

    class RMSNorm : public Mln::Sdk::Module {
        public:
            RMSNorm(int64_t dim, double eps, bool zero_centered = false);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &x);

        private:
            int64_t dim_;
            double eps_;
            bool zero_centered_;
    };
}
