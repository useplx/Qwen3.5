#include "qwen3_5/nn/lm_head.hpp"
#include "qwen3_5/config.hpp"
#include "qwen3_5/nn/rmsnorm.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/operations.hpp>

using namespace Mln::Sdk;
using namespace Qwen35::Arch;

namespace Qwen35::Nn {
    LMHead::LMHead() { name = "model_head"; }

    Tensor LMHead::forward(const Tensor &hidden) {
        Tensor selected = hidden;
        if (hidden.shape.size() == 2) {
            const int64_t sequence = hidden.shape[0];
            selected = operations.insert<Functional::Slice>(
                hidden, std::vector<int64_t>{sequence - 1, 0},
                std::vector<int64_t>{sequence, HIDDEN_SIZE}, std::vector<int64_t>{1, 1});
            selected = operations.insert<Functional::Reshape>(std::move(selected),
                                                              std::vector<int64_t>{HIDDEN_SIZE});
        }
        Tensor norm_weight = operations.insert<Functional::Const>(
            Weight{"model.language_model.norm.weight", {HIDDEN_SIZE}}, true);
        Tensor normed =
            rms_norm(operations, selected, norm_weight, HIDDEN_SIZE, RMS_NORM_EPS, true);

        Tensor weight = operations.insert<Functional::Const>(
            Weight{"model.language_model.embed_tokens.weight", {VOCAB_SIZE, HIDDEN_SIZE}}, true);
        Tensor weight_t =
            operations.insert<Functional::Transpose>(std::move(weight), std::vector<int64_t>{1, 0});
        Tensor logits =
            operations.insert<Functional::MatMul>(std::move(normed), std::move(weight_t));
        return operations.insert<Functional::Convert>(std::move(logits), Mln::DType::Float32);
    }
}
