#pragma once

#include "qwen3_vl/errors.hpp"

#include <mln/sdk/native/serialization.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace Qwen3VL {
    struct Text {
            std::string value;
    };

    struct Image {
            std::vector<std::uint8_t> encoded;
    };

    struct Video {
            std::vector<Image> frames;
            float frames_per_second = 2.0F;
    };

    using Content = std::variant<Text, Image, Video>;

    struct Input {
            std::vector<Content> content;
    };

    inline constexpr std::uint32_t InputMagic = 0x51333549;

}

namespace Mln::Sdk::Native {
    template <> struct Return<Qwen3VL::Text> {
            static void write(Return<void>::Writer &writer, const Qwen3VL::Text &text) {
                writer.write(text.value);
            }
    };

    template <> struct Argument<Qwen3VL::Text> {
            static Qwen3VL::Text read(Argument<void>::Reader &reader) {
                return Qwen3VL::Text{reader.read<std::string>()};
            }
    };

    template <> struct Return<Qwen3VL::Image> {
            static void write(Return<void>::Writer &writer, const Qwen3VL::Image &image) {
                writer.write(image.encoded);
            }
    };

    template <> struct Argument<Qwen3VL::Image> {
            static Qwen3VL::Image read(Argument<void>::Reader &reader) {
                return Qwen3VL::Image{reader.read<std::vector<std::uint8_t>>()};
            }
    };

    template <> struct Return<Qwen3VL::Video> {
            static void write(Return<void>::Writer &writer, const Qwen3VL::Video &video) {
                writer.write_field(video.frames);
                writer.write(video.frames_per_second);
            }
    };

    template <> struct Argument<Qwen3VL::Video> {
            static Qwen3VL::Video read(Argument<void>::Reader &reader) {
                Qwen3VL::Video video;
                video.frames = reader.read_field<std::vector<Qwen3VL::Image>>();
                video.frames_per_second = reader.read<float>();
                return video;
            }
    };

    template <> struct Return<Qwen3VL::Input> {
            static void write(Return<void>::Writer &writer, const Qwen3VL::Input &input) {
                writer.write(Qwen3VL::InputMagic);
                writer.write(static_cast<std::uint32_t>(input.content.size()));
                for (const Qwen3VL::Content &content : input.content) {
                    writer.write(static_cast<std::uint8_t>(content.index()));
                    std::visit([&](const auto &value) { writer.write_field(value); }, content);
                }
            }
    };

    template <> struct Argument<Qwen3VL::Input> {
            static Qwen3VL::Input read(Argument<void>::Reader &reader) {
                const std::uint32_t magic = reader.read<std::uint32_t>();
                if (magic != Qwen3VL::InputMagic)
                    Mln::raise(Qwen3VL::Errors::Input::Magic::Invalid, magic);
                Qwen3VL::Input input;
                const std::uint32_t count = reader.read<std::uint32_t>();
                input.content.reserve(count);
                for (std::uint32_t index = 0; index < count; ++index) {
                    const std::uint8_t kind = reader.read<std::uint8_t>();
                    switch (kind) {
                    case 0:
                        input.content.emplace_back(reader.read_field<Qwen3VL::Text>());
                        break;
                    case 1:
                        input.content.emplace_back(reader.read_field<Qwen3VL::Image>());
                        break;
                    case 2:
                        input.content.emplace_back(reader.read_field<Qwen3VL::Video>());
                        break;
                    default:
                        Mln::raise(Qwen3VL::Errors::Input::Content::Kind::Invalid, kind, index);
                    }
                }
                return input;
            }
    };

}
