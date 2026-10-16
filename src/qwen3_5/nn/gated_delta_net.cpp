#include "qwen3_5/nn/gated_delta_net.hpp"

#include "qwen3_5/config.hpp"
#include "qwen3_5/nn/linear.hpp"
#include "qwen3_5/nn/rmsnorm.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/operations.hpp>

#include <cmath>
#include <utility>
#include <vector>

using namespace Mln::Sdk;
using namespace Qwen35::Arch;

namespace Qwen35::Nn {
    namespace {
        Tensor scalar(Operations &operations, double value, const std::vector<int64_t> &shape,
                      Mln::DType dtype = Mln::DType::Float32) {
            Tensor constant =
                operations.insert<Functional::Const>(Weight{"", {}, value}, false, dtype);
            return operations.insert<Functional::Broadcast>(std::move(constant), shape,
                                                            std::vector<int64_t>{});
        }

        Tensor slice(Operations &operations, const Tensor &value, int64_t begin, int64_t end) {
            return operations.insert<Functional::Slice>(value, std::vector<int64_t>{begin},
                                                        std::vector<int64_t>{end},
                                                        std::vector<int64_t>{1});
        }

        Tensor columns(Operations &operations, const Tensor &value, int64_t rows, int64_t begin,
                       int64_t end) {
            return operations.insert<Functional::Slice>(value, std::vector<int64_t>{0, begin},
                                                        std::vector<int64_t>{rows, end},
                                                        std::vector<int64_t>{1, 1});
        }

    }

    void GatedDeltaNet::Output::apply_hook(Module &module, std::span<const Tensor> inputs) {
        hidden = module.apply(std::move(hidden), inputs, "hidden");
        convolution = module.apply(std::move(convolution), inputs, "convolution");
        recurrent = module.apply(std::move(recurrent), inputs, "recurrent");
    }

    GatedDeltaNet::GatedDeltaNet() {
        name = "linear_attn";
        modules.insert<Linear>("in_proj_qkv", HIDDEN_SIZE, LINEAR_CONV_DIM);
        modules.insert<Linear>("in_proj_z", HIDDEN_SIZE, LINEAR_VALUE_SIZE);
        modules.insert<Linear>("in_proj_b", HIDDEN_SIZE, LINEAR_NUM_HEADS);
        modules.insert<Linear>("in_proj_a", HIDDEN_SIZE, LINEAR_NUM_HEADS);
        modules.insert<Linear>("out_proj", LINEAR_VALUE_SIZE, HIDDEN_SIZE);
        parameters.insert("conv1d.weight", {{}, {LINEAR_CONV_DIM, 1, LINEAR_CONV_KERNEL}});
        parameters.insert("A_log", {{}, {LINEAR_NUM_HEADS}});
        parameters.insert("dt_bias", {{}, {LINEAR_NUM_HEADS}});
        parameters.insert("norm.weight", {{}, {LINEAR_HEAD_DIM}});
    }

    GatedDeltaNet::Output GatedDeltaNet::forward(const Tensor &hidden, const Tensor &convolution,
                                                 const Tensor &recurrent) {
        const int64_t sequence = hidden.shape.size() == 1 ? 1 : hidden.shape[0];
        Tensor mixed = modules.forward<Linear>("in_proj_qkv", hidden);
        Tensor z = modules.forward<Linear>("in_proj_z", hidden);
        Tensor beta = modules.forward<Linear>("in_proj_b", hidden);
        Tensor step = modules.forward<Linear>("in_proj_a", hidden);
        mixed = operations.insert<Functional::Reshape>(
            std::move(mixed), std::vector<int64_t>{sequence, LINEAR_CONV_DIM});
        z = operations.insert<Functional::Reshape>(
            std::move(z), std::vector<int64_t>{sequence, LINEAR_VALUE_SIZE});
        beta = operations.insert<Functional::Reshape>(
            std::move(beta), std::vector<int64_t>{sequence, LINEAR_NUM_HEADS});
        step = operations.insert<Functional::Reshape>(
            std::move(step), std::vector<int64_t>{sequence, LINEAR_NUM_HEADS});

        Tensor conv_weight = parameters.at("conv1d.weight");
        conv_weight = operations.insert<Functional::Reshape>(
            std::move(conv_weight), std::vector<int64_t>{LINEAR_CONV_DIM, LINEAR_CONV_KERNEL});
        Tensor a_log = parameters.at("A_log");
        Tensor dt_bias = parameters.at("dt_bias");
        Tensor norm_weight = parameters.at("norm.weight");
        a_log = operations.insert<Functional::Convert>(std::move(a_log), Mln::DType::Float32);
        dt_bias = operations.insert<Functional::Convert>(std::move(dt_bias), Mln::DType::Float32);

        mixed = operations.insert<Functional::Convert>(std::move(mixed), Mln::DType::Float32);
        z = operations.insert<Functional::Convert>(std::move(z), Mln::DType::Float32);
        beta = operations.insert<Functional::Convert>(std::move(beta), Mln::DType::Float32);
        step = operations.insert<Functional::Convert>(std::move(step), Mln::DType::Float32);
        conv_weight =
            operations.insert<Functional::Convert>(std::move(conv_weight), Mln::DType::Float32);
        Tensor initial_convolution =
            operations.insert<Functional::Convert>(convolution, Mln::DType::Float32);
        constexpr int64_t CONVOLUTION_ELEMENTS = LINEAR_CONV_DIM * (LINEAR_CONV_KERNEL - 1);
        constexpr int64_t RECURRENT_ELEMENTS = LINEAR_NUM_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM;
        const int64_t OUTPUT_ELEMENTS = sequence * LINEAR_VALUE_SIZE;

        Tensor convolution_result = operations.insert<Functional::CausalConv1D>(
            std::move(mixed), std::move(conv_weight), std::move(initial_convolution));
        Tensor activated = slice(operations, convolution_result, 0, sequence * LINEAR_CONV_DIM);
        activated = operations.insert<Functional::Reshape>(
            std::move(activated), std::vector<int64_t>{sequence, LINEAR_CONV_DIM});
        Tensor next_convolution = slice(operations, convolution_result, sequence * LINEAR_CONV_DIM,
                                        sequence * LINEAR_CONV_DIM + CONVOLUTION_ELEMENTS);
        next_convolution = operations.insert<Functional::Reshape>(
            std::move(next_convolution),
            std::vector<int64_t>{LINEAR_CONV_DIM, LINEAR_CONV_KERNEL - 1});
        next_convolution = operations.insert<Functional::Convert>(std::move(next_convolution),
                                                                  Mln::DType::Float16);

        Tensor query = columns(operations, activated, sequence, 0, LINEAR_KEY_SIZE);
        Tensor key = columns(operations, activated, sequence, LINEAR_KEY_SIZE, 2 * LINEAR_KEY_SIZE);
        Tensor value =
            columns(operations, activated, sequence, 2 * LINEAR_KEY_SIZE, LINEAR_CONV_DIM);
        query = operations.insert<Functional::Reshape>(
            std::move(query), std::vector<int64_t>{sequence, LINEAR_NUM_HEADS, LINEAR_HEAD_DIM});
        key = operations.insert<Functional::Reshape>(
            std::move(key), std::vector<int64_t>{sequence, LINEAR_NUM_HEADS, LINEAR_HEAD_DIM});
        value = operations.insert<Functional::Reshape>(
            std::move(value), std::vector<int64_t>{sequence, LINEAR_NUM_HEADS, LINEAR_HEAD_DIM});

        beta = operations.insert<Functional::Logistic>(std::move(beta));
        dt_bias = operations.insert<Functional::Broadcast>(
            std::move(dt_bias), std::vector<int64_t>{sequence, LINEAR_NUM_HEADS},
            std::vector<int64_t>{1});
        step = operations.insert<Functional::Add>(std::move(step), std::move(dt_bias));
        step = operations.insert<Functional::Softplus>(std::move(step));
        a_log = operations.insert<Functional::Exp>(std::move(a_log));
        a_log = operations.insert<Functional::Broadcast>(
            std::move(a_log), std::vector<int64_t>{sequence, LINEAR_NUM_HEADS},
            std::vector<int64_t>{1});
        Tensor decay = operations.insert<Functional::Multiply>(std::move(a_log), std::move(step));
        decay = operations.insert<Functional::Subtract>(
            scalar(operations, 0.0, {sequence, LINEAR_NUM_HEADS}), std::move(decay));

        Tensor delta_result = operations.insert<Functional::GatedDeltaRule>(
            std::move(query), std::move(key), std::move(value), std::move(decay), std::move(beta),
            operations.insert<Functional::Convert>(recurrent, Mln::DType::Float32));
        Tensor outputs = slice(operations, delta_result, 0, OUTPUT_ELEMENTS);
        outputs = operations.insert<Functional::Reshape>(
            std::move(outputs), std::vector<int64_t>{sequence * LINEAR_NUM_HEADS, LINEAR_HEAD_DIM});
        Tensor next_recurrent =
            slice(operations, delta_result, OUTPUT_ELEMENTS, OUTPUT_ELEMENTS + RECURRENT_ELEMENTS);
        next_recurrent = operations.insert<Functional::Reshape>(
            std::move(next_recurrent),
            std::vector<int64_t>{LINEAR_NUM_HEADS, LINEAR_HEAD_DIM, LINEAR_HEAD_DIM});

        z = operations.insert<Functional::Reshape>(
            std::move(z), std::vector<int64_t>{sequence * LINEAR_NUM_HEADS, LINEAR_HEAD_DIM});
        outputs = rms_norm(operations, outputs, norm_weight, LINEAR_HEAD_DIM, RMS_NORM_EPS,
                           Mln::DType::Float32, false);
        Tensor gate =
            operations.insert<Functional::Multiply>(z, operations.insert<Functional::Logistic>(z));
        outputs = operations.insert<Functional::Multiply>(std::move(outputs), std::move(gate));
        outputs = operations.insert<Functional::Reshape>(
            std::move(outputs), std::vector<int64_t>{sequence, LINEAR_VALUE_SIZE});
        outputs = operations.insert<Functional::Convert>(std::move(outputs), Mln::DType::Float16);
        outputs = modules.forward<Linear>("out_proj", outputs);
        if (hidden.shape.size() == 1)
            outputs = operations.insert<Functional::Reshape>(std::move(outputs),
                                                             std::vector<int64_t>{HIDDEN_SIZE});
        return {std::move(outputs), std::move(next_convolution), std::move(next_recurrent)};
    }
}
