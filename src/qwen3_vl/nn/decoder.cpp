#include "qwen3_vl/nn/decoder.hpp"
#include "qwen3_vl/errors.hpp"

#include "qwen3_vl/config.hpp"
#include "qwen3_vl/nn/decoder_layer.hpp"
#include "qwen3_vl/nn/embedding.hpp"
#include "qwen3_vl/nn/lm_head.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/operations.hpp>

#include <algorithm>
#include <string>
#include <utility>

using namespace Mln::Sdk;
using namespace Qwen3VL::Arch;

namespace Qwen3VL::Nn {
    namespace {
        int64_t cache_elements(const Config &config) {
            return NUM_KV_HEADS * config.max_seq_len * HEAD_DIM;
        }

        Tensor slice_state(Operations &operations, const Tensor &state, int64_t offset,
                           int64_t elements, std::vector<int64_t> shape) {
            Tensor flat = operations.insert<Functional::Slice>(
                state, std::vector<int64_t>{offset}, std::vector<int64_t>{offset + elements},
                std::vector<int64_t>{1});
            return operations.insert<Functional::Reshape>(std::move(flat), std::move(shape));
        }

        Tensor flatten(Operations &operations, Tensor state, int64_t elements) {
            return operations.insert<Functional::Reshape>(std::move(state),
                                                          std::vector<int64_t>{elements});
        }

        Tensor int_scalar(Operations &operations, int64_t value) {
            return operations.insert<Functional::Const>(Weight{"", {}, static_cast<double>(value)},
                                                        false, Mln::DType::Int32);
        }

        Tensor update_state(Operations &operations, Tensor state, Tensor value, int64_t offset,
                            int64_t elements) {
            Tensor flat = flatten(operations, std::move(value), elements);
            return operations.insert<Functional::DynamicUpdateSlice>(
                std::move(state), std::move(flat),
                std::vector<Tensor>{int_scalar(operations, offset)});
        }
    }

    int64_t Decoder::storage_elements(const Config &config) {
        const int64_t full_layers = NUM_LAYERS / FULL_ATTENTION_INTERVAL;
        const int64_t linear_layers = NUM_LAYERS - full_layers;
        return full_layers * 2 * cache_elements(config) +
               linear_layers * LINEAR_CONV_DIM * (LINEAR_CONV_KERNEL - 1);
    }

    int64_t Decoder::recurrent_elements() {
        const int64_t linear_layers = NUM_LAYERS - NUM_LAYERS / FULL_ATTENTION_INTERVAL;
        return linear_layers * LINEAR_NUM_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM;
    }

    Decoder::Decoder(Config config, Output output) : config_(config), output_(output) {
        config_.validate();
        name = "model";
        modules.insert<Embedding>("language_model.embed_tokens");
        for (int64_t layer = 0; layer < NUM_LAYERS; ++layer) {
            modules.insert<Decoder::Layer>("language_model.layers." + std::to_string(layer), layer);
        }
        if (output_ == Output::Logits)
            modules.insert<LMHead>("language_model.head");
    }

    std::string Decoder::entrypoint() const {
        return output_ == Output::Logits ? Module::entrypoint() : "model.prefill.forward";
    }

    std::vector<Mln::Sdk::Tensor> Decoder::graph(std::span<const Mln::Sdk::Tensor> inputs) {
        if (inputs.size() != 10)
            Mln::raise(Errors::Decoder::Graph::Input::Count::Invalid, inputs.size());
        if (output_ == Output::State) {
            Transition state = transition(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                                          inputs[5], inputs[6], inputs[7], inputs[8], inputs[9]);
            return {std::move(state.storage), std::move(state.recurrent)};
        }
        auto [logits, storage, recurrent] =
            forward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
                    inputs[7], inputs[8], inputs[9]);
        return {std::move(logits), std::move(storage), std::move(recurrent)};
    }

    std::vector<std::vector<Tensor>> Decoder::specializations() const {
        std::vector<std::vector<Tensor>> result;
        const auto append = [&](int64_t sequence) {
            result.push_back(
                {Tensor({sequence}, Mln::DType::Float32), Tensor({storage_elements(config_)}),
                 Tensor({recurrent_elements()}, Mln::DType::Float32),
                 Tensor({sequence, ROTARY_DIM}), Tensor({sequence, ROTARY_DIM}),
                 Tensor(std::vector<int64_t>{}), Tensor({sequence, config_.max_seq_len}),
                 Tensor({Tensor::Dynamic, HIDDEN_SIZE}), Tensor({sequence}, Mln::DType::Float32),
                 Tensor({sequence, 1})});
        };

        append(1);
        if (output_ == Output::Logits)
            return result;
        for (int64_t sequence = config_.prefill_chunk; sequence >= config_.prefill_min_chunk;
             sequence /= 2)
            append(sequence);
        return result;
    }

    std::tuple<Tensor, Tensor, Tensor>
    Decoder::forward(const Tensor &token, const Tensor &storage, const Tensor &recurrent,
                     const Tensor &cos, const Tensor &sin, const Tensor &position,
                     const Tensor &mask, const Tensor &visual_embeddings,
                     const Tensor &visual_indices, const Tensor &visual_mask) {
        auto [hidden, next_storage, next_recurrent] =
            transition(token, storage, recurrent, cos, sin, position, mask, visual_embeddings,
                       visual_indices, visual_mask);
        const int64_t sequence = token.shape.empty() ? 1 : token.shape[0];
        Tensor last_hidden = operations.insert<Functional::Slice>(
            std::move(hidden), std::vector<int64_t>{sequence - 1, 0},
            std::vector<int64_t>{sequence, HIDDEN_SIZE}, std::vector<int64_t>{1, 1});
        last_hidden = operations.insert<Functional::Reshape>(std::move(last_hidden),
                                                             std::vector<int64_t>{HIDDEN_SIZE});
        Tensor logits = modules.forward<LMHead>("language_model.head", last_hidden);
        return {std::move(logits), std::move(next_storage), std::move(next_recurrent)};
    }

    Decoder::Transition Decoder::transition(const Tensor &token, const Tensor &storage,
                                            const Tensor &recurrent, const Tensor &cos,
                                            const Tensor &sin, const Tensor &position,
                                            const Tensor &mask, const Tensor &visual_embeddings,
                                            const Tensor &visual_indices,
                                            const Tensor &visual_mask) {
        const int64_t sequence = token.shape.empty() ? 1 : token.shape[0];
        Tensor hidden = modules.forward<Embedding>("language_model.embed_tokens", token);
        Tensor indices = operations.insert<Functional::Convert>(visual_indices, Mln::DType::Int32);
        Tensor selected =
            operations.insert<Functional::Gather>(visual_embeddings, std::move(indices), 0);
        Tensor expanded_mask = operations.insert<Functional::Broadcast>(
            visual_mask, std::vector<int64_t>{sequence, HIDDEN_SIZE}, std::vector<int64_t>{0, 1});
        hidden = operations.insert<Functional::Add>(
            hidden, operations.insert<Functional::Multiply>(
                        operations.insert<Functional::Subtract>(std::move(selected), hidden),
                        std::move(expanded_mask)));
        Tensor next_storage = storage;
        Tensor next_recurrent = recurrent;

        int64_t storage_offset = 0;
        int64_t recurrent_offset = 0;
        for (int64_t layer = 0; layer < NUM_LAYERS; ++layer) {
            const bool full_attention = (layer + 1) % FULL_ATTENTION_INTERVAL == 0;
            Tensor state_a;
            Tensor state_b;
            int64_t state_a_elements;
            int64_t state_b_elements;
            const int64_t state_a_offset = storage_offset;
            int64_t state_b_offset;
            if (full_attention) {
                state_a_elements = cache_elements(config_);
                state_b_elements = cache_elements(config_);
                state_a = slice_state(operations, storage, storage_offset, state_a_elements,
                                      {NUM_KV_HEADS, config_.max_seq_len, HEAD_DIM});
                storage_offset += state_a_elements;
                state_b = slice_state(operations, storage, storage_offset, state_b_elements,
                                      {NUM_KV_HEADS, config_.max_seq_len, HEAD_DIM});
                state_b_offset = storage_offset;
                storage_offset += state_b_elements;
            } else {
                state_a_elements = LINEAR_CONV_DIM * (LINEAR_CONV_KERNEL - 1);
                state_b_elements = LINEAR_NUM_HEADS * LINEAR_HEAD_DIM * LINEAR_HEAD_DIM;
                state_a = slice_state(operations, storage, storage_offset, state_a_elements,
                                      {LINEAR_CONV_DIM, LINEAR_CONV_KERNEL - 1});
                storage_offset += state_a_elements;
                state_b = slice_state(operations, recurrent, recurrent_offset, state_b_elements,
                                      {LINEAR_NUM_HEADS, LINEAR_HEAD_DIM, LINEAR_HEAD_DIM});
                state_b_offset = recurrent_offset;
                recurrent_offset += state_b_elements;
            }
            Decoder::Layer::Output output =
                modules.forward<Decoder::Layer>("language_model.layers." + std::to_string(layer),
                                                hidden, cos, sin, state_a, state_b, position, mask);
            hidden = std::move(output.hidden);

            next_storage =
                update_state(operations, std::move(next_storage), std::move(output.state_a),
                             state_a_offset, state_a_elements);
            if (full_attention) {
                next_storage =
                    update_state(operations, std::move(next_storage), std::move(output.state_b),
                                 state_b_offset, state_b_elements);
            } else {
                next_recurrent =
                    update_state(operations, std::move(next_recurrent), std::move(output.state_b),
                                 state_b_offset, state_b_elements);
            }
        }

        return {std::move(hidden), std::move(next_storage), std::move(next_recurrent)};
    }
}
