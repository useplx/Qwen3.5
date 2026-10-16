#pragma once

#include "qwen3_5/errors.hpp"

#include <mln/sdk/native/serialization.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace Qwen35 {
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
    template <> struct Return<Qwen35::Text> {
            static void write(Return<void>::Writer &writer, const Qwen35::Text &text) {
                writer.write(text.value);
            }
    };

    template <> struct Argument<Qwen35::Text> {
            static Qwen35::Text read(Argument<void>::Reader &reader) {
                return Qwen35::Text{reader.read<std::string>()};
            }
    };

    template <> struct Return<Qwen35::Image> {
            static void write(Return<void>::Writer &writer, const Qwen35::Image &image) {
                writer.write(image.encoded);
            }
    };

    template <> struct Argument<Qwen35::Image> {
            static Qwen35::Image read(Argument<void>::Reader &reader) {
                return Qwen35::Image{reader.read<std::vector<std::uint8_t>>()};
            }
    };

    template <> struct Return<Qwen35::Video> {
            static void write(Return<void>::Writer &writer, const Qwen35::Video &video) {
                writer.write_field(video.frames);
                writer.write(video.frames_per_second);
            }
    };

    template <> struct Argument<Qwen35::Video> {
            static Qwen35::Video read(Argument<void>::Reader &reader) {
                Qwen35::Video video;
                video.frames = reader.read_field<std::vector<Qwen35::Image>>();
                video.frames_per_second = reader.read<float>();
                return video;
            }
    };

    template <> struct Return<Qwen35::Input> {
            static void write(Return<void>::Writer &writer, const Qwen35::Input &input) {
                writer.write(Qwen35::InputMagic);
                writer.write(static_cast<std::uint32_t>(input.content.size()));
                for (const Qwen35::Content &content : input.content) {
                    writer.write(static_cast<std::uint8_t>(content.index()));
                    std::visit([&](const auto &value) { writer.write_field(value); }, content);
                }
            }
    };

    template <> struct Argument<Qwen35::Input> {
            static Qwen35::Input read(Argument<void>::Reader &reader) {
                const std::uint32_t magic = reader.read<std::uint32_t>();
                if (magic != Qwen35::InputMagic)
                    Mln::raise(Qwen35::Errors::Input::Magic::Invalid, magic);
                Qwen35::Input input;
                const std::uint32_t count = reader.read<std::uint32_t>();
                input.content.reserve(count);
                for (std::uint32_t index = 0; index < count; ++index) {
                    const std::uint8_t kind = reader.read<std::uint8_t>();
                    switch (kind) {
                    case 0:
                        input.content.emplace_back(reader.read_field<Qwen35::Text>());
                        break;
                    case 1:
                        input.content.emplace_back(reader.read_field<Qwen35::Image>());
                        break;
                    case 2:
                        input.content.emplace_back(reader.read_field<Qwen35::Video>());
                        break;
                    default:
                        Mln::raise(Qwen35::Errors::Input::Content::Kind::Invalid, kind, index);
                    }
                }
                return input;
            }
    };

}
