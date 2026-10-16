#include "qwen3_5/nn/decoder_layer.hpp"
#include "qwen3_5/config.hpp"
#include "qwen3_5/nn/attention.hpp"
#include "qwen3_5/nn/gated_delta_net.hpp"
#include "qwen3_5/nn/mlp.hpp"
#include "qwen3_5/nn/rmsnorm.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/operations.hpp>

#include <string>

using namespace Mln::Sdk;
using namespace Qwen35::Arch;

namespace Qwen35::Nn {
    Decoder::Layer::Layer(int64_t layer_idx)
        : full_attention_((layer_idx + 1) % FULL_ATTENTION_INTERVAL == 0) {
        name = "model.language_model.layers." + std::to_string(layer_idx);
        modules.insert<RMSNorm>("input_layernorm", HIDDEN_SIZE, RMS_NORM_EPS, true);
        if (full_attention_)
            modules.insert<Attention>("self_attn");
        else
            modules.insert<GatedDeltaNet>("linear_attn");
        modules.insert<RMSNorm>("post_attention_layernorm", HIDDEN_SIZE, RMS_NORM_EPS, true);
        modules.insert<MLP>("mlp");
    }

    Decoder::Layer::Output Decoder::Layer::forward(const Tensor &hidden, const Tensor &cos,
                                                   const Tensor &sin, const Tensor &k_cache,
                                                   const Tensor &v_cache, const Tensor &position,
                                                   const Tensor &mask) {
        Tensor residual = hidden;
        Tensor h = modules.forward<RMSNorm>("input_layernorm", hidden);
        Tensor mixed;
        Tensor state_a;
        Tensor state_b;
        if (full_attention_) {
            Attention::Output attention = modules.forward<Attention>(
                "self_attn", h, cos, sin, k_cache, v_cache, position, mask);
            mixed = std::move(attention.hidden);
            state_a = std::move(attention.k_cache);
            state_b = std::move(attention.v_cache);
        } else {
            GatedDeltaNet::Output delta =
                modules.forward<GatedDeltaNet>("linear_attn", h, k_cache, v_cache);
            mixed = std::move(delta.hidden);
            state_a = std::move(delta.convolution);
            state_b = std::move(delta.recurrent);
        }
        Tensor after_attn =
            operations.insert<Functional::Add>(std::move(residual), std::move(mixed));

        residual = after_attn;
        h = modules.forward<RMSNorm>("post_attention_layernorm", after_attn);
        h = modules.forward<MLP>("mlp", h);
        Tensor final_hidden = operations.insert<Functional::Add>(std::move(residual), std::move(h));

        return Decoder::Layer::Output{
            std::move(final_hidden),
            std::move(state_a),
            std::move(state_b),
        };
    }
}
