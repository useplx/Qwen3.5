#include "qwen3_5/model.hpp"
#include "qwen3_5/config.hpp"
#include "qwen3_5/errors.hpp"
#include "qwen3_5/float16.hpp"
#include "qwen3_5/nn/decoder.hpp"
#include "qwen3_5/rope.hpp"

#include <mln/sdk/native.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <print>
#include <vector>

using namespace Mln::Sdk;
using namespace Qwen35::Arch;

namespace Qwen35 {
    IMPORT(qwen35, tokenizer, Mln::Buffer(uint32_t, Mln::Buffer, const Mln::Buffer &));

    namespace {
        Mln::Buffer float_buffer(const float *values, std::size_t count) {
            Mln::Buffer buffer = Mln::Buffer(static_cast<uint32_t>(count * sizeof(float)));
            if (count != 0)
                buffer.write(std::as_bytes(std::span(values, count)));
            return buffer;
        }

        Mln::Buffer half_buffer(const float *values, std::size_t count) {
            std::vector<uint16_t> halves(count);
            for (std::size_t i = 0; i < count; ++i)
                halves[i] = to_float16(values[i]);
            Mln::Buffer buffer =
                Mln::Buffer(static_cast<uint32_t>(halves.size() * sizeof(uint16_t)));
            if (!halves.empty())
                buffer.write(std::as_bytes(std::span(halves)));
            return buffer;
        }

        std::vector<float> read_floats(Mln::Buffer buffer, int64_t count) {
            std::vector<float> values(count);
            buffer.read(std::as_writable_bytes(std::span(values)));
            return values;
        }

        uint64_t next_random(uint64_t &state) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            return state * 0x2545f4914f6cdd1dULL;
        }

        bool is_generation_control_token(int64_t token) {
            return token == VISION_START_TOKEN_ID || token == VISION_END_TOKEN_ID ||
                   token == VISION_PAD_TOKEN_ID || token == IMAGE_TOKEN_ID ||
                   token == VIDEO_TOKEN_ID || token == IM_START_TOKEN_ID;
        }

        struct Position {
                int64_t temporal;
                int64_t height;
                int64_t width;
        };

        struct Positions {
                std::vector<Position> values;
                int64_t delta = 0;
        };

        Positions mrope_positions(std::span<const float> tokens,
                                  std::span<const Vision::Grid> grids) {
            Positions result;
            result.values.reserve(tokens.size());
            std::vector<Vision::Grid> expanded_grids;
            for (const Vision::Grid &grid : grids) {
                const std::uint32_t count = grid.kind == Vision::Kind::Video ? grid.temporal : 1;
                for (std::uint32_t index = 0; index < count; ++index) {
                    Vision::Grid frame = grid;
                    frame.temporal = 1;
                    expanded_grids.push_back(frame);
                }
            }
            std::size_t visual_index = 0;
            int64_t current = 0;
            for (std::size_t offset = 0; offset < tokens.size();) {
                const int64_t token = static_cast<int64_t>(tokens[offset]);
                if (token != IMAGE_TOKEN_ID && token != VIDEO_TOKEN_ID) {
                    result.values.push_back({current, current, current});
                    ++current;
                    ++offset;
                    continue;
                }

                const std::size_t begin = offset;
                while (offset < tokens.size() && static_cast<int64_t>(tokens[offset]) == token)
                    ++offset;
                const std::size_t count = offset - begin;
                if (visual_index >= expanded_grids.size())
                    Mln::raise(Errors::Vision::Grid::Missing, visual_index);
                const Vision::Grid &grid = expanded_grids[visual_index++];
                const int64_t height = grid.height / VISION_SPATIAL_MERGE_SIZE;
                const int64_t width = grid.width / VISION_SPATIAL_MERGE_SIZE;
                const std::size_t expected = static_cast<std::size_t>(height * width);
                if (count != expected)
                    Mln::raise(Errors::Vision::Token::Count::Invalid, count, expected);
                for (int64_t index = 0; index < static_cast<int64_t>(count); ++index)
                    result.values.push_back(
                        {current, current + index / width, current + index % width});
                current += std::max(height, width);
            }
            if (visual_index != expanded_grids.size())
                Mln::raise(Errors::Vision::Grid::Unused, expanded_grids.size() - visual_index);
            result.delta = current - static_cast<int64_t>(tokens.size());
            return result;
        }

        void apply_repetition_penalty(std::vector<float> &logits, std::span<const float> prompt,
                                      std::span<const float> generated, double penalty) {
            if (penalty == 1.0)
                return;
            std::vector<bool> adjusted(logits.size());
            const auto apply = [&](std::span<const float> tokens) {
                for (float encoded : tokens) {
                    const int64_t token = static_cast<int64_t>(encoded);
                    if (token < 0 || token >= static_cast<int64_t>(logits.size()) ||
                        adjusted[static_cast<std::size_t>(token)])
                        continue;
                    float &logit = logits[static_cast<std::size_t>(token)];
                    logit = logit < 0.0F ? static_cast<float>(logit * penalty)
                                         : static_cast<float>(logit / penalty);
                    adjusted[static_cast<std::size_t>(token)] = true;
                }
            };
            apply(prompt);
            apply(generated);
        }

        void apply_presence_penalty(std::vector<float> &logits, std::span<const float> generated,
                                    double penalty) {
            if (penalty == 0.0)
                return;
            std::vector<bool> adjusted(logits.size());
            for (float encoded : generated) {
                const int64_t token = static_cast<int64_t>(encoded);
                if (token < 0 || token >= static_cast<int64_t>(logits.size()) ||
                    adjusted[static_cast<std::size_t>(token)])
                    continue;
                logits[static_cast<std::size_t>(token)] -= static_cast<float>(penalty);
                adjusted[static_cast<std::size_t>(token)] = true;
            }
        }

        int64_t greedy_token(const std::vector<float> &logits) {
            int64_t selected = 0;
            float maximum = -std::numeric_limits<float>::infinity();
            for (int64_t token = 0; token < static_cast<int64_t>(logits.size()); ++token) {
                if (is_generation_control_token(token) || !std::isfinite(logits[token]))
                    continue;
                if (logits[token] > maximum) {
                    maximum = logits[token];
                    selected = token;
                }
            }
            return selected;
        }

        int64_t sample_token(const std::vector<float> &logits, const Config &config,
                             uint64_t &random_state) {
            struct Candidate {
                    int64_t token;
                    float logit;
            };
            std::vector<Candidate> candidates;
            candidates.reserve(logits.size());
            for (int64_t token = 0; token < static_cast<int64_t>(logits.size()); ++token) {
                if (is_generation_control_token(token))
                    continue;
                const float value = std::isfinite(logits[token])
                                        ? logits[token]
                                        : -std::numeric_limits<float>::infinity();
                candidates.push_back(Candidate{token, value});
            }
            const int64_t count = config.top_k == 0
                                      ? static_cast<int64_t>(candidates.size())
                                      : std::min<int64_t>(config.top_k, candidates.size());
            std::partial_sort(candidates.begin(), candidates.begin() + count, candidates.end(),
                              [](const Candidate &left, const Candidate &right) {
                                  return left.logit > right.logit;
                              });
            candidates.resize(static_cast<std::size_t>(count));

            const double maximum = candidates.front().logit;
            std::vector<double> weights(candidates.size());
            double total = 0.0;
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                weights[i] = std::exp((candidates[i].logit - maximum) / config.temperature);
                total += weights[i];
            }
            if (!(total > 0.0) || !std::isfinite(total))
                return candidates.front().token;

            double kept = 0.0;
            std::size_t cutoff = 0;
            do {
                kept += weights[cutoff++];
            } while (cutoff < weights.size() && kept / total < config.top_p);

            const double unit = static_cast<double>(next_random(random_state) >> 11) /
                                static_cast<double>(uint64_t{1} << 53);
            const double selected = unit * kept;
            double cumulative = 0.0;
            for (std::size_t i = 0; i < cutoff; ++i) {
                cumulative += weights[i];
                if (selected < cumulative)
                    return candidates[i].token;
            }
            return candidates[cutoff - 1].token;
        }
    }

    Mln::Buffer Model::tokenize(Mln::Buffer text, const Mln::Buffer &tokenizer_asset) {
        return tokenizer(0, std::move(text), tokenizer_asset);
    }

    Mln::Buffer Model::tokenize_chat(Mln::Buffer system_and_user,
                                     const Mln::Buffer &tokenizer_asset) {
        return tokenizer(2, std::move(system_and_user), tokenizer_asset);
    }

    Mln::Buffer Model::tokenize_visual(std::span<const Content> contents,
                                       std::span<const Vision::Grid> grids,
                                       const Mln::Buffer &tokenizer_asset) {
        std::string prompt;
        std::size_t visual_index = 0;
        const std::size_t visual_count =
            static_cast<std::size_t>(std::ranges::count_if(contents, [](const Content &content) {
                return !std::holds_alternative<Text>(content);
            }));
        if (visual_count != grids.size())
            Mln::raise(Errors::Vision::Grid::Content::Mismatch::Invalid, visual_count,
                       grids.size());
        for (const Content &content : contents) {
            if (const auto *text = std::get_if<Text>(&content)) {
                prompt += text->value;
                continue;
            }
            const Vision::Grid &grid = grids[visual_index++];
            const std::size_t frame_tokens =
                static_cast<std::size_t>(grid.height / VISION_SPATIAL_MERGE_SIZE) *
                (grid.width / VISION_SPATIAL_MERGE_SIZE);
            const std::uint32_t frames = grid.kind == Vision::Kind::Video ? grid.temporal : 1;
            for (std::uint32_t frame = 0; frame < frames; ++frame) {
                if (grid.kind == Vision::Kind::Video)
                    prompt += std::format("<{:.1f} seconds>", (static_cast<float>(frame) + 0.25F) *
                                                                  grid.seconds_per_grid);
                prompt += "<|vision_start|>";
                const std::string_view pad =
                    grid.kind == Vision::Kind::Image ? "<|image_pad|>" : "<|video_pad|>";
                for (std::size_t token = 0; token < frame_tokens; ++token)
                    prompt += pad;
                prompt += "<|vision_end|>";
            }
        }
        Mln::Buffer content(prompt.size());
        if (!prompt.empty())
            content.write(std::as_bytes(std::span(prompt)));
        return tokenizer(3, std::move(content), tokenizer_asset);
    }

    Mln::Buffer Model::detokenize(Mln::Buffer tokens, const Mln::Buffer &tokenizer_asset) {
        return tokenizer(1, std::move(tokens), tokenizer_asset);
    }

    Mln::Buffer Model::generate(Mln::Runtime::Session &session, Mln::Buffer prompt_tokens,
                                const Mln::Buffer &tokenizer_asset) {
        const std::size_t bytes = static_cast<std::size_t>(HIDDEN_SIZE) * sizeof(std::uint16_t);
        Mln::Runtime::Session::State visual =
            session.state("vision.embeddings", Mln::Buffer(bytes));
        return generate(session, std::move(prompt_tokens), tokenizer_asset, visual.data(), {});
    }

    Mln::Buffer Model::generate(Mln::Runtime::Session &session, Mln::Buffer prompt_tokens,
                                const Mln::Buffer &tokenizer_asset,
                                const Mln::Buffer &visual_embeddings,
                                std::span<const Vision::Grid> grids) {
        Mln::Buffer owned_prompt_tokens = std::move(prompt_tokens);
        const int64_t prompt_len = static_cast<int64_t>(owned_prompt_tokens.size() / sizeof(float));
        if (prompt_len <= 0 || prompt_len >= config_.max_seq_len)
            Mln::raise(Errors::Decoder::Prompt::Token::Count::Invalid, prompt_len,
                       config_.max_seq_len);
        if (config_.top_k < 0)
            Mln::raise(Errors::Decoder::Sampling::TopK::Invalid, config_.top_k);
        if (!(config_.top_p > 0.0 && config_.top_p <= 1.0))
            Mln::raise(Errors::Decoder::Sampling::TopP::Invalid, config_.top_p);
        if (!(config_.temperature > 0.0))
            Mln::raise(Errors::Decoder::Sampling::Temperature::Invalid, config_.temperature);
        if (!(config_.repetition_penalty > 0.0))
            Mln::raise(Errors::Decoder::Sampling::Repetition::Penalty::Invalid,
                       config_.repetition_penalty);
        if (config_.presence_penalty < 0.0)
            Mln::raise(Errors::Decoder::Sampling::Presence::Penalty::Invalid,
                       config_.presence_penalty);
        std::vector<float> prompt(prompt_len);
        owned_prompt_tokens.read(std::as_writable_bytes(std::span(prompt)));
        const Positions positions = mrope_positions(prompt, grids);

        const uint32_t storage_bytes =
            static_cast<uint32_t>(Nn::Decoder::storage_elements(config_) * sizeof(uint16_t));
        const uint32_t recurrent_bytes =
            static_cast<uint32_t>(Nn::Decoder::recurrent_elements() * sizeof(float));
        Mln::Runtime::Session::State decoder_storage =
            session.state("decoder.storage", Mln::Buffer(storage_bytes));
        Mln::Runtime::Session::State decoder_recurrent =
            session.state("decoder.recurrent", Mln::Buffer(recurrent_bytes));

        std::vector<float> generated;
        uint32_t emitted_text_bytes = 0;
        uint64_t random_state = 0xcbf29ce484222325ULL;
        for (float token : prompt) {
            random_state ^= static_cast<uint64_t>(token);
            random_state *= 0x100000001b3ULL;
        }
        if (random_state == 0)
            random_state = 1;
        int64_t current_token = prompt_len > 0 ? static_cast<int64_t>(prompt[0]) : BOS_TOKEN_ID;
        std::size_t next_visual_token = 0;

        for (int64_t pos = 0; pos < config_.max_seq_len - 1; pos++) {
            const int64_t remaining = prompt_len - pos;
            const int64_t prefillable = remaining - 1;
            int64_t sequence = 1;
            if (prefillable >= config_.prefill_min_chunk) {
                sequence = config_.prefill_min_chunk;
                while (sequence * 2 <= config_.prefill_chunk && sequence * 2 <= prefillable)
                    sequence *= 2;
            }
            const int64_t context = config_.max_seq_len;
            std::vector<float> tokens(static_cast<std::size_t>(sequence));
            std::vector<float> cos(static_cast<std::size_t>(sequence * ROTARY_DIM));
            std::vector<float> sin(static_cast<std::size_t>(sequence * ROTARY_DIM));
            std::vector<float> mask(static_cast<std::size_t>(sequence * context));
            std::vector<float> visual_indices(static_cast<std::size_t>(sequence));
            std::vector<float> visual_mask(static_cast<std::size_t>(sequence));
            for (int64_t item = 0; item < sequence; ++item) {
                tokens[item] =
                    sequence == 1 ? static_cast<float>(current_token) : prompt[pos + item];
                if (static_cast<int64_t>(tokens[item]) == IMAGE_TOKEN_ID ||
                    static_cast<int64_t>(tokens[item]) == VIDEO_TOKEN_ID) {
                    visual_indices[item] = static_cast<float>(next_visual_token++);
                    visual_mask[item] = 1.0f;
                }
                const int64_t absolute_position = pos + item;
                const Position rope_position =
                    absolute_position < prompt_len
                        ? positions.values[static_cast<std::size_t>(absolute_position)]
                        : Position{absolute_position + positions.delta,
                                   absolute_position + positions.delta,
                                   absolute_position + positions.delta};
                rope_cos_sin(rope_position.temporal, rope_position.height, rope_position.width,
                             cos.data() + item * ROTARY_DIM, sin.data() + item * ROTARY_DIM);
                causal_mask(absolute_position, context, mask.data() + item * context);
            }

            const float pos_f = static_cast<float>(pos);
            std::array<Mln::Buffer, 7> owned_decoder_inputs{
                Mln::Buffer(float_buffer(tokens.data(), tokens.size())),
                Mln::Buffer(half_buffer(cos.data(), cos.size())),
                Mln::Buffer(half_buffer(sin.data(), sin.size())),
                Mln::Buffer(half_buffer(&pos_f, 1)),
                Mln::Buffer(half_buffer(mask.data(), mask.size())),
                Mln::Buffer(float_buffer(visual_indices.data(), visual_indices.size())),
                Mln::Buffer(half_buffer(visual_mask.data(), visual_mask.size())),
            };
            std::array<std::reference_wrapper<const Mln::Buffer>, 10> decoder_inputs{
                owned_decoder_inputs[0], decoder_storage.data(),  decoder_recurrent.data(),
                owned_decoder_inputs[1], owned_decoder_inputs[2], owned_decoder_inputs[3],
                owned_decoder_inputs[4], visual_embeddings,       owned_decoder_inputs[5],
                owned_decoder_inputs[6],
            };
            const bool prefill = remaining > 1;
            std::vector<Mln::Buffer> decoder_outputs =
                session.invoke(prefill ? "model.prefill.forward" : "model.forward", decoder_inputs);
            int64_t next_token;
            if (prefill) {
                if (decoder_outputs.size() != 2)
                    Mln::raise(Errors::Decoder::Prefill::Runtime::Output::Count::Invalid,
                               decoder_outputs.size());
                decoder_storage = session.state("decoder.storage", std::move(decoder_outputs[0]));
                decoder_recurrent =
                    session.state("decoder.recurrent", std::move(decoder_outputs[1]));
                next_token = static_cast<int64_t>(prompt[pos + sequence]);
            } else {
                if (decoder_outputs.size() != 3)
                    Mln::raise(Errors::Decoder::Runtime::Output::Count::Invalid,
                               decoder_outputs.size());
                std::vector<float> logits = read_floats(std::move(decoder_outputs[0]), VOCAB_SIZE);
                decoder_storage = session.state("decoder.storage", std::move(decoder_outputs[1]));
                decoder_recurrent =
                    session.state("decoder.recurrent", std::move(decoder_outputs[2]));
                apply_repetition_penalty(logits, prompt, generated, config_.repetition_penalty);
                apply_presence_penalty(logits, generated, config_.presence_penalty);
                next_token = config_.do_sample ? sample_token(logits, config_, random_state)
                                               : greedy_token(logits);
                generated.push_back(static_cast<float>(next_token));
                Mln::Buffer partial_tokens(generated.size() * sizeof(float));
                partial_tokens.write(std::as_bytes(std::span(generated)));
                Mln::Buffer partial_text = tokenizer(1, std::move(partial_tokens), tokenizer_asset);
                if (partial_text.size() > emitted_text_bytes) {
                    session.emit(partial_text, emitted_text_bytes);
                    emitted_text_bytes = partial_text.size();
                }
                if (next_token == EOS_TOKEN_ID || next_token == END_OF_TEXT_TOKEN_ID ||
                    next_token == IM_END_TOKEN_ID ||
                    static_cast<int64_t>(generated.size()) >= config_.max_new_tokens)
                    break;
            }
            current_token = next_token;
            pos += sequence - 1;
        }

        std::size_t expected_visual_tokens = 0;
        for (const Vision::Grid &grid : grids)
            expected_visual_tokens += static_cast<std::size_t>(grid.temporal) * grid.height *
                                      grid.width /
                                      (VISION_SPATIAL_MERGE_SIZE * VISION_SPATIAL_MERGE_SIZE);
        if (next_visual_token != expected_visual_tokens)
            Mln::raise(Errors::Vision::Token::Count::Invalid, next_visual_token,
                       expected_visual_tokens);

        Mln::Buffer result(generated.size() * sizeof(float));
        if (!generated.empty())
            result.write(std::as_bytes(std::span(generated)));
        return tokenizer(1, std::move(result), tokenizer_asset);
    }
}
