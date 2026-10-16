#include "qwen3_5/nn/mlp.hpp"
#include "qwen3_5/config.hpp"
#include "qwen3_5/nn/linear.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/nn/silu.hpp>
#include <mln/sdk/operations.hpp>

using namespace Mln::Sdk;
using namespace Qwen35::Arch;

namespace Qwen35::Nn {
    MLP::MLP() {
        name = "mlp";
        modules.insert<Linear>("gate_proj", HIDDEN_SIZE, INTERMEDIATE_SIZE);
        modules.insert<Linear>("up_proj", HIDDEN_SIZE, INTERMEDIATE_SIZE);
        modules.insert<Linear>("down_proj", INTERMEDIATE_SIZE, HIDDEN_SIZE);
        modules.insert<Mln::Sdk::Nn::SiLU>("act");
    }

    Tensor MLP::forward(const Tensor &x) {
        Tensor gate = modules.forward<Linear>("gate_proj", x);
        gate = modules.forward<Mln::Sdk::Nn::SiLU>("act", gate);
        Tensor up = modules.forward<Linear>("up_proj", x);
        Tensor hidden = operations.insert<Functional::Multiply>(std::move(gate), std::move(up));
        return modules.forward<Linear>("down_proj", hidden);
    }
}
