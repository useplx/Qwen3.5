#include "qwen3_5/nn/vision.hpp"

#include "qwen3_5/config.hpp"
#include "qwen3_5/errors.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/nn/gelu.hpp>
#include <mln/sdk/operations.hpp>

#include <cmath>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

using namespace Mln::Sdk;
using namespace Qwen35::Arch;

namespace Qwen35::Nn::Vision {
    namespace {
        Tensor scalar(Operations &operations, double value, const std::vector<std::int64_t> &shape,
                      const Tensor &shape_source = {},
                      std::optional<Mln::DType> dtype = std::nullopt) {
            Tensor constant =
                dtype ? operations.insert<Functional::Const>(Weight{"", {}, value}, false, *dtype)
                : shape_source.dtype
                    ? operations.insert<Functional::Const>(Weight{"", {}, value}, false,
                                                           *shape_source.dtype)
                    : operations.insert<Functional::Const>(Weight{"", {}, value}, true);
            if (std::ranges::find(shape, Tensor::Dynamic) != shape.end())
                return operations.insert<Functional::Broadcast>(
                    std::move(constant), shape, std::vector<std::int64_t>{}, shape_source);
            return operations.insert<Functional::Broadcast>(std::move(constant), shape,
                                                            std::vector<std::int64_t>{});
        }

        Tensor tile(Operations &operations, Tensor value, const std::vector<std::int64_t> &shape,
                    std::vector<std::int64_t> dimensions) {
            return operations.insert<Functional::Broadcast>(std::move(value), shape,
                                                            std::move(dimensions));
        }

        Tensor rotate_half(Operations &operations, const Tensor &value) {
            const std::int64_t patches = value.shape[1];
            Tensor first = operations.insert<Functional::Slice>(
                value, std::vector<std::int64_t>{0, 0, 0},
                std::vector<std::int64_t>{VISION_NUM_HEADS, patches, HeadDim / 2},
                std::vector<std::int64_t>{1, 1, 1}, value, std::vector<std::int64_t>{1});
            Tensor second = operations.insert<Functional::Slice>(
                value, std::vector<std::int64_t>{0, 0, HeadDim / 2},
                std::vector<std::int64_t>{VISION_NUM_HEADS, patches, HeadDim},
                std::vector<std::int64_t>{1, 1, 1}, value, std::vector<std::int64_t>{1});
            Tensor negative_one = operations.insert<Functional::Broadcast>(
                operations.insert<Functional::Const>(Weight{"", {}, -1.0}, false,
                                                     Mln::DType::Float32),
                std::vector<std::int64_t>{VISION_NUM_HEADS, patches, HeadDim / 2},
                std::vector<std::int64_t>{}, value);
            second =
                operations.insert<Functional::Multiply>(std::move(second), std::move(negative_one));
            return operations.insert<Functional::Concatenate>(std::move(second), std::move(first),
                                                              2);
        }

        Tensor apply_rotary(Operations &operations, const Tensor &value, const Tensor &cos,
                            const Tensor &sin) {
            const std::int64_t patches = value.shape[1];
            Tensor computation = operations.insert<Functional::Convert>(value, Mln::DType::Float32);
            Tensor cosine = tile(operations, cos, {VISION_NUM_HEADS, patches, HeadDim}, {1, 2});
            Tensor sine = tile(operations, sin, {VISION_NUM_HEADS, patches, HeadDim}, {1, 2});
            Tensor rotated = operations.insert<Functional::Add>(
                operations.insert<Functional::Multiply>(computation, std::move(cosine)),
                operations.insert<Functional::Multiply>(rotate_half(operations, computation),
                                                        std::move(sine)));
            return operations.insert<Functional::Convert>(
                std::move(rotated), value.dtype.value_or(Mln::DType::Float16));
        }
    }

    Dense::Dense(std::int64_t inputs, std::int64_t outputs) : inputs_(inputs), outputs_(outputs) {
        parameters.insert("weight", {{}, {outputs, inputs}});
        parameters.insert("bias", {{}, {outputs}});
    }

    Tensor Dense::forward(const Tensor &input) {
        Tensor weight = parameters.at("weight");
        weight = operations.insert<Functional::Transpose>(std::move(weight),
                                                          std::vector<std::int64_t>{1, 0});
        Tensor projected = operations.insert<Functional::MatMul>(input, std::move(weight));
        Tensor bias = operations.insert<Functional::Broadcast>(
            parameters.at("bias"), projected.shape,
            std::vector<std::int64_t>{static_cast<std::int64_t>(projected.shape.size() - 1)},
            projected);
        return operations.insert<Functional::Add>(std::move(projected), std::move(bias));
    }

    LayerNorm::LayerNorm(std::int64_t width) : width_(width) {
        parameters.insert("weight", {{}, {width}});
        parameters.insert("bias", {{}, {width}});
    }

    Tensor LayerNorm::forward(const Tensor &input) {
        const std::int64_t rows = input.shape.front();
        const Mln::DType output_dtype = input.dtype.value_or(Mln::DType::Float16);

        Tensor computation = operations.insert<Functional::Convert>(input, Mln::DType::Float32);
        Tensor inverse = scalar(operations, 1.0 / static_cast<double>(width_), {rows, width_},
                                computation, Mln::DType::Float32);
        Tensor mean = operations.insert<Functional::ReduceSum>(
            operations.insert<Functional::Multiply>(computation, inverse),
            std::vector<std::int64_t>{1});
        Tensor centered = operations.insert<Functional::Subtract>(
            computation, tile(operations, std::move(mean), {rows, width_}, {0}));
        Tensor variance = operations.insert<Functional::ReduceSum>(
            operations.insert<Functional::Multiply>(
                operations.insert<Functional::Multiply>(centered, centered), std::move(inverse)),
            std::vector<std::int64_t>{1});
        Tensor epsilon = scalar(operations, 1e-6, {rows}, variance, Mln::DType::Float32);
        Tensor deviation = operations.insert<Functional::Rsqrt>(
            operations.insert<Functional::Add>(std::move(variance), std::move(epsilon)));
        Tensor normalized = operations.insert<Functional::Multiply>(
            std::move(centered), tile(operations, std::move(deviation), {rows, width_}, {0}));
        Tensor result = operations.insert<Functional::Add>(
            operations.insert<Functional::Multiply>(
                std::move(normalized), operations.insert<Functional::Broadcast>(
                                           operations.insert<Functional::Convert>(
                                               parameters.at("weight"), Mln::DType::Float32),
                                           std::vector<std::int64_t>{rows, width_},
                                           std::vector<std::int64_t>{1}, computation)),
            operations.insert<Functional::Broadcast>(
                operations.insert<Functional::Convert>(parameters.at("bias"), Mln::DType::Float32),
                std::vector<std::int64_t>{rows, width_}, std::vector<std::int64_t>{1},
                computation));
        return operations.insert<Functional::Convert>(std::move(result), output_dtype);
    }

    Attention::Attention() {
        modules.insert<Dense>("qkv", VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE * 3);
        modules.insert<Dense>("proj", VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE);
    }

    Tensor Attention::forward(const Tensor &hidden, const Tensor &cos, const Tensor &sin,
                              const Tensor &segments) {
        return stages(hidden, cos, sin, segments).back();
    }

    std::vector<Tensor> Attention::stages(const Tensor &hidden, const Tensor &cos,
                                          const Tensor &sin, const Tensor &segments) {
        const std::int64_t patches = hidden.shape[0];
        Tensor qkv = modules.forward<Dense>("qkv", hidden);
        Tensor flat_qkv = qkv;
        qkv = operations.insert<Functional::Reshape>(
            std::move(qkv), std::vector<std::int64_t>{patches, 3, VISION_NUM_HEADS, HeadDim});
        auto component = [&](std::int64_t index) {
            Tensor value = operations.insert<Functional::Slice>(
                qkv, std::vector<std::int64_t>{0, index, 0, 0},
                std::vector<std::int64_t>{patches, index + 1, VISION_NUM_HEADS, HeadDim},
                std::vector<std::int64_t>{1, 1, 1, 1}, qkv, std::vector<std::int64_t>{0});
            value = operations.insert<Functional::Reshape>(
                std::move(value), std::vector<std::int64_t>{patches, VISION_NUM_HEADS, HeadDim});
            return operations.insert<Functional::Transpose>(std::move(value),
                                                            std::vector<std::int64_t>{1, 0, 2});
        };
        Tensor query = apply_rotary(operations, component(0), cos, sin);
        Tensor key = apply_rotary(operations, component(1), cos, sin);
        Tensor value = component(2);
        Tensor rotated_query = query;
        Tensor rotated_key = key;
        Tensor attention_value = value;

        Tensor row_segments = operations.insert<Functional::Broadcast>(
            segments, std::vector<std::int64_t>{patches, patches}, std::vector<std::int64_t>{0},
            hidden);
        Tensor column_segments = operations.insert<Functional::Broadcast>(
            segments, std::vector<std::int64_t>{patches, patches}, std::vector<std::int64_t>{1},
            hidden);
        Tensor visible = operations.insert<Functional::Equal>(std::move(row_segments),
                                                              std::move(column_segments));
        Tensor mask = operations.insert<Functional::Select>(
            std::move(visible), scalar(operations, 0.0, {patches, patches}, hidden),
            scalar(operations, -1.0e4, {patches, patches}, hidden));
        Tensor context = operations.insert<Functional::Attention>(
            std::move(query), std::move(key), std::move(value),
            1.0 / std::sqrt(static_cast<double>(HeadDim)), std::move(mask));
        context = operations.insert<Functional::Transpose>(std::move(context),
                                                           std::vector<std::int64_t>{1, 0, 2});
        context = operations.insert<Functional::Reshape>(
            std::move(context), std::vector<std::int64_t>{patches, VISION_HIDDEN_SIZE});
        Tensor flat_context = context;
        Tensor projected = modules.forward<Dense>("proj", context);
        return {std::move(flat_qkv),        std::move(rotated_query), std::move(rotated_key),
                std::move(attention_value), std::move(flat_context),  std::move(projected)};
    }

    Block::Block(std::int64_t) {
        modules.insert<LayerNorm>("norm1", VISION_HIDDEN_SIZE);
        modules.insert<Attention>("attn");
        modules.insert<LayerNorm>("norm2", VISION_HIDDEN_SIZE);
        modules.insert<Dense>("mlp.linear_fc1", VISION_HIDDEN_SIZE, VISION_INTERMEDIATE_SIZE);
        modules.insert<Mln::Sdk::Nn::GELU>("mlp.activation");
        modules.insert<Dense>("mlp.linear_fc2", VISION_INTERMEDIATE_SIZE, VISION_HIDDEN_SIZE);
    }

    Tensor Block::forward(const Tensor &hidden, const Tensor &cos, const Tensor &sin,
                          const Tensor &segments) {
        return stages(hidden, cos, sin, segments).second;
    }

    std::pair<Tensor, Tensor> Block::stages(const Tensor &hidden, const Tensor &cos,
                                            const Tensor &sin, const Tensor &segments) {
        Tensor attended = modules.forward<Attention>(
            "attn", modules.forward<LayerNorm>("norm1", hidden), cos, sin, segments);
        Tensor residual = operations.insert<Functional::Add>(hidden, std::move(attended));
        Tensor expanded =
            modules.forward<Dense>("mlp.linear_fc1", modules.forward<LayerNorm>("norm2", residual));
        expanded = modules.forward<Mln::Sdk::Nn::GELU>("mlp.activation", expanded);
        Tensor projected = modules.forward<Dense>("mlp.linear_fc2", expanded);
        Tensor output = operations.insert<Functional::Add>(residual, std::move(projected));
        return {std::move(residual), std::move(output)};
    }

    std::vector<Tensor> Block::attention_stages(const Tensor &hidden, const Tensor &cos,
                                                const Tensor &sin, const Tensor &segments) {
        Tensor normalized = modules.forward<LayerNorm>("norm1", hidden);
        std::vector<Tensor> outputs{normalized};
        auto attention = modules.at<Attention>("attn")->stages(normalized, cos, sin, segments);
        outputs.insert(outputs.end(), std::make_move_iterator(attention.begin()),
                       std::make_move_iterator(attention.end()));
        outputs.push_back(operations.insert<Functional::Add>(hidden, outputs.back()));
        return outputs;
    }

    Merger::Merger() {
        modules.insert<LayerNorm>("norm", VISION_HIDDEN_SIZE);
        modules.insert<Dense>("linear_fc1", VISION_HIDDEN_SIZE * 4, VISION_HIDDEN_SIZE * 4);
        modules.insert<Mln::Sdk::Nn::GELU>("activation");
        modules.insert<Dense>("linear_fc2", VISION_HIDDEN_SIZE * 4, VISION_OUT_HIDDEN_SIZE);
    }

    Tensor Merger::forward(const Tensor &hidden) {
        const std::int64_t merged_tokens =
            hidden.shape[0] == Tensor::Dynamic ? Tensor::Dynamic : hidden.shape[0] / 4;
        Tensor merged = modules.forward<LayerNorm>("norm", hidden);
        merged = operations.insert<Functional::Reshape>(
            std::move(merged), std::vector<std::int64_t>{merged_tokens, 4, VISION_HIDDEN_SIZE});
        merged = operations.insert<Functional::Reshape>(
            std::move(merged), std::vector<std::int64_t>{merged_tokens, VISION_HIDDEN_SIZE * 4});
        merged = modules.forward<Dense>("linear_fc1", merged);
        merged = modules.forward<Mln::Sdk::Nn::GELU>("activation", merged);
        return modules.forward<Dense>("linear_fc2", merged);
    }

    Encoder::Encoder(Config config) : config_(config) {
        config_.validate();
        name = "model.visual";
        for (std::int64_t layer = 0; layer < VISION_DEPTH; ++layer)
            modules.insert<Block>("blocks." + std::to_string(layer), layer);
        modules.insert<Merger>("merger");
    }

    std::vector<Tensor> Encoder::graph(std::span<const Tensor> inputs) {
        if (inputs.size() != 6)
            Mln::raise(Errors::Vision::Graph::Input::Count::Invalid, inputs.size());
#if defined(QWEN35_VISION_STAGE_PROBE)
        return stages(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]);
#else
        return {forward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])};
#endif
    }

    std::vector<std::vector<Tensor>> Encoder::specializations() const {
        const auto inputs = [](std::int64_t patches) {
            return std::vector<Tensor>{
                Tensor({patches, PatchElements}, Mln::DType::Float16),
                Tensor({patches, 4}, Mln::DType::Float32),
                Tensor({patches, 4}, Mln::DType::Float32),
                Tensor({patches, HeadDim}, Mln::DType::Float32),
                Tensor({patches, HeadDim}, Mln::DType::Float32),
                Tensor({patches}, Mln::DType::Int32),
            };
        };

        std::vector<std::vector<Tensor>> result;
        for (std::int64_t patches = 256; patches <= config_.vision_max_patches; patches *= 2)
            result.push_back(inputs(patches));
        return result;
    }

    Tensor Encoder::forward(const Tensor &patches, const Tensor &position_indices,
                            const Tensor &position_weights, const Tensor &cos, const Tensor &sin,
                            const Tensor &segments) {
        return stages(patches, position_indices, position_weights, cos, sin, segments).back();
    }

    std::vector<Tensor> Encoder::stages(const Tensor &patches, const Tensor &position_indices,
                                        const Tensor &position_weights, const Tensor &cos,
                                        const Tensor &sin, const Tensor &segments) {
        const std::int64_t patch_count = patches.shape[0];
        Tensor patch_weight = operations.insert<Functional::Const>(
            Weight{"model.visual.patch_embed.proj.weight",
                   {VISION_HIDDEN_SIZE, VISION_IN_CHANNELS, VISION_TEMPORAL_PATCH_SIZE,
                    VISION_PATCH_SIZE, VISION_PATCH_SIZE}},
            true);
        patch_weight = operations.insert<Functional::Reshape>(
            std::move(patch_weight), std::vector<std::int64_t>{VISION_HIDDEN_SIZE, PatchElements});
        patch_weight = operations.insert<Functional::Transpose>(std::move(patch_weight),
                                                                std::vector<std::int64_t>{1, 0});
        Tensor hidden = operations.insert<Functional::MatMul>(patches, std::move(patch_weight));
        Tensor patch_bias = operations.insert<Functional::Const>(
            Weight{"model.visual.patch_embed.proj.bias", {VISION_HIDDEN_SIZE}}, true);
        Tensor expanded_bias = operations.insert<Functional::Broadcast>(
            std::move(patch_bias), std::vector<std::int64_t>{patch_count, VISION_HIDDEN_SIZE},
            std::vector<std::int64_t>{1}, hidden);
        hidden = operations.insert<Functional::Add>(std::move(hidden), std::move(expanded_bias));

        Tensor position_table = operations.insert<Functional::Const>(
            Weight{"model.visual.pos_embed.weight", {2304, VISION_HIDDEN_SIZE}}, true);
        Tensor indices =
            operations.insert<Functional::Convert>(position_indices, Mln::DType::Int32);
        Tensor positions =
            operations.insert<Functional::Gather>(std::move(position_table), std::move(indices), 0);
        positions =
            operations.insert<Functional::Convert>(std::move(positions), Mln::DType::Float32);
        Tensor weights =
            tile(operations, position_weights, {patch_count, 4, VISION_HIDDEN_SIZE}, {0, 1});
        positions = operations.insert<Functional::ReduceSum>(
            operations.insert<Functional::Multiply>(std::move(positions), std::move(weights)),
            std::vector<std::int64_t>{1});
        positions = operations.insert<Functional::Convert>(
            std::move(positions), hidden.dtype.value_or(Mln::DType::Float16));
        hidden = operations.insert<Functional::Add>(std::move(hidden), std::move(positions));

        std::vector<Tensor> outputs;
        outputs.reserve(static_cast<std::size_t>(VISION_DEPTH * 2 + 2));
        outputs.push_back(hidden);
        for (std::int64_t layer = 0; layer < VISION_DEPTH; ++layer) {
#if defined(QWEN35_VISION_STAGE_PROBE)
            if (layer == 0) {
                auto attention =
                    modules.at<Block>("blocks.0")->attention_stages(hidden, cos, sin, segments);
                outputs.insert(outputs.end(), std::make_move_iterator(attention.begin()),
                               std::make_move_iterator(attention.end()));
            }
#endif
            auto [attention, output] = modules.at<Block>("blocks." + std::to_string(layer))
                                           ->stages(hidden, cos, sin, segments);
            outputs.push_back(std::move(attention));
            hidden = std::move(output);
            outputs.push_back(hidden);
        }
        outputs.push_back(modules.forward<Merger>("merger", hidden));
        return outputs;
    }
}
