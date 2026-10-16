#include "qwen3_vl/nn/embedding.hpp"
#include "qwen3_vl/config.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/operations.hpp>

using namespace Mln::Sdk;
using namespace Qwen3VL::Arch;

namespace Qwen3VL::Nn {
    Embedding::Embedding() { name = "model_embed"; }

    Tensor Embedding::forward(const Tensor &token_id) {
        Tensor idx = operations.insert<Functional::Convert>(token_id, Mln::DType::Int32);
        Tensor table = operations.insert<Functional::Const>(
            Weight{"model.language_model.embed_tokens.weight", {VOCAB_SIZE, HIDDEN_SIZE}}, true);
        return operations.insert<Functional::Gather>(std::move(table), std::move(idx), 0);
    }
}
