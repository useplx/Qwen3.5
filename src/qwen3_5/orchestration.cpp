#include "qwen3_5/orchestration.hpp"
#include "qwen3_5/config.hpp"
#include "qwen3_5/errors.hpp"
#include "qwen3_5/float16.hpp"
#include "qwen3_5/input.hpp"
#include "qwen3_5/vision.hpp"

#include <mln/sdk/native.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace Mln::Sdk;

namespace Qwen35 {
    IMPORT(qwen35, vision, Vision::Input(Vision::Request));

    namespace {
        template <typename T> Mln::Buffer buffer(const std::vector<T> &values) {
            static_assert(std::is_trivially_copyable_v<T>);
            Mln::Buffer result(values.size() * sizeof(T));
            if (!values.empty())
                result.write(std::as_bytes(std::span(values)));
            return result;
        }

        Input read_input(const Mln::Buffer &value) {
            std::vector<std::byte> bytes(value.size());
            value.read(bytes);
            Mln::Sdk::Native::Argument<void>::Reader reader(bytes);
            Input input = reader.read<Input>();
            if (!reader.empty())
                Mln::raise(Mln::Sdk::Errors::Native::Serialization::Field::Bytes::Trailing,
                           reader.remaining());
            return input;
        }

        void pad_visual_input(Vision::Input &input) {
            const std::size_t patches = input.segments.size();
            constexpr std::size_t maximum = QWEN35_VISION_MAX_PATCHES;
            std::size_t bucket = 256;
            while (bucket < patches && bucket < maximum)
                bucket *= 2;
            if (patches > maximum)
                Mln::raise(Errors::Config::Vision::Patches::Exceeded, patches, maximum);
            if (bucket == patches)
                return;

            constexpr std::size_t patch_elements =
                Arch::VISION_IN_CHANNELS * Arch::VISION_TEMPORAL_PATCH_SIZE *
                Arch::VISION_PATCH_SIZE * Arch::VISION_PATCH_SIZE;
            constexpr std::size_t head_dimension =
                Arch::VISION_HIDDEN_SIZE / Arch::VISION_NUM_HEADS;
            input.patches.resize(bucket * patch_elements);
            input.position_indices.resize(bucket * 4);
            input.position_weights.resize(bucket * 4);
            input.cos.resize(bucket * head_dimension);
            input.sin.resize(bucket * head_dimension);
            input.segments.resize(bucket, -1);
        }
    }

    Mln::Buffer Orchestration::forward(Mln::Runtime::Session &session, Mln::Buffer input) {
        Mln::Buffer tokenizer_asset = session.asset("tokenizer.json");
        std::array<std::byte, sizeof(std::uint32_t)> prefix{};
        if (input.size() >= prefix.size())
            input.read(0, prefix);
        std::uint32_t magic = 0;
        std::memcpy(&magic, prefix.data(), sizeof(magic));
        if (magic == InputMagic) {
            Input request = read_input(input);
            const bool has_media = std::ranges::any_of(request.content, [](const Content &item) {
                return !std::holds_alternative<Text>(item);
            });
            if (!has_media) {
                std::string text;
                for (const Content &item : request.content)
                    text += std::get<Text>(item).value;
                Mln::Buffer text_buffer(text.size());
                if (!text.empty())
                    text_buffer.write(std::as_bytes(std::span(text)));
                return generate(session, tokenize(std::move(text_buffer), tokenizer_asset),
                                tokenizer_asset);
            }
            Vision::Request vision_request;
            for (Content &item : request.content) {
                if (auto *image = std::get_if<Image>(&item))
                    vision_request.media.emplace_back(std::move(*image));
                else if (auto *video = std::get_if<Video>(&item))
                    vision_request.media.emplace_back(std::move(*video));
            }
            Vision::Input prepared = vision(std::move(vision_request));
            pad_visual_input(prepared);
            std::array<Mln::Buffer, 6> vision_inputs{
                buffer(prepared.patches),
                buffer(prepared.position_indices),
                buffer(prepared.position_weights),
                buffer(prepared.cos),
                buffer(prepared.sin),
                buffer(prepared.segments),
            };
            std::vector<Mln::Buffer> visual = session.invoke("model.visual.forward", vision_inputs);
            if (visual.size() != 1)
                Mln::raise(Errors::Vision::Runtime::Output::Count::Invalid, visual.size());
            Mln::Runtime::Session::State embeddings =
                session.state("vision.embeddings", std::move(visual.front()));

            Mln::Buffer prompt_tokens =
                tokenize_visual(request.content, prepared.grids, tokenizer_asset);
            return generate(session, std::move(prompt_tokens), tokenizer_asset, embeddings.data(),
                            prepared.grids);
        }

        std::vector<std::byte> payload(input.size());
        input.read(payload);
        const std::string_view raw(reinterpret_cast<const char *>(payload.data()), payload.size());
        const bool chat = std::ranges::find(payload, std::byte{}) != payload.end();
        Mln::Buffer prompt_tokens = chat ? tokenize_chat(std::move(input), tokenizer_asset)
                                         : tokenize(std::move(input), tokenizer_asset);
        return generate(session, std::move(prompt_tokens), tokenizer_asset);
    }
}
