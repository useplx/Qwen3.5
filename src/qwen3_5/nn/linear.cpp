#include "qwen3_5/nn/linear.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/operations.hpp>

using namespace Mln::Sdk;

namespace Qwen35::Nn {
    Linear::Linear(int64_t in_features, int64_t out_features)
        : in_features_(in_features), out_features_(out_features) {
        parameters.insert("weight", {{}, {out_features, in_features}});
    }

    Tensor Linear::forward(const Tensor &x) {
        Tensor weight = parameters.at("weight");
        Tensor transposed_weight =
            operations.insert<Functional::Transpose>(std::move(weight), std::vector<int64_t>{1, 0});
        return operations.insert<Functional::MatMul>(x, std::move(transposed_weight));
    }
}
