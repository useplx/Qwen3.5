#pragma once

#include "qwen3_vl/input.hpp"

#include <mln/sdk/native/serialization.hpp>

#include <cstdint>
#include <variant>
#include <vector>

namespace Qwen3VL::Vision {
    enum class Kind : std::uint32_t {
        Image,
        Video,
    };

    struct Grid {
            Kind kind = Kind::Image;
            std::uint32_t temporal = 1;
            std::uint32_t height = 0;
            std::uint32_t width = 0;
            float seconds_per_grid = 0.0F;
    };

    struct Request {
            using Media = std::variant<Qwen3VL::Image, Qwen3VL::Video>;
            std::vector<Media> media;
    };

    struct Input {
            std::vector<std::uint16_t> patches;
            std::vector<float> position_indices;
            std::vector<float> position_weights;
            std::vector<float> cos;
            std::vector<float> sin;
            std::vector<std::int32_t> segments;
            std::vector<Grid> grids;
    };
}

namespace Mln::Sdk::Native {
    template <> struct Return<Qwen3VL::Vision::Grid> {
            static void write(Return<void>::Writer &writer, const Qwen3VL::Vision::Grid &grid) {
                writer.write(grid.kind);
                writer.write(grid.temporal);
                writer.write(grid.height);
                writer.write(grid.width);
                writer.write(grid.seconds_per_grid);
            }
    };

    template <> struct Argument<Qwen3VL::Vision::Grid> {
            static Qwen3VL::Vision::Grid read(Argument<void>::Reader &reader) {
                return {reader.read<Qwen3VL::Vision::Kind>(), reader.read<std::uint32_t>(),
                        reader.read<std::uint32_t>(), reader.read<std::uint32_t>(),
                        reader.read<float>()};
            }
    };

    template <> struct Return<Qwen3VL::Vision::Request> {
            static void write(Return<void>::Writer &writer,
                              const Qwen3VL::Vision::Request &request) {
                writer.write(static_cast<std::uint32_t>(request.media.size()));
                for (const Qwen3VL::Vision::Request::Media &media : request.media) {
                    writer.write(static_cast<std::uint8_t>(media.index()));
                    std::visit([&](const auto &value) { writer.write_field(value); }, media);
                }
            }
    };

    template <> struct Argument<Qwen3VL::Vision::Request> {
            static Qwen3VL::Vision::Request read(Argument<void>::Reader &reader) {
                Qwen3VL::Vision::Request request;
                const std::uint32_t count = reader.read<std::uint32_t>();
                request.media.reserve(count);
                for (std::uint32_t index = 0; index < count; ++index) {
                    const std::uint8_t kind = reader.read<std::uint8_t>();
                    switch (kind) {
                    case 0:
                        request.media.emplace_back(reader.read_field<Qwen3VL::Image>());
                        break;
                    case 1:
                        request.media.emplace_back(reader.read_field<Qwen3VL::Video>());
                        break;
                    default:
                        Mln::raise(Qwen3VL::Errors::Input::Content::Kind::Invalid, kind, index);
                    }
                }
                return request;
            }
    };

    template <> struct Return<Qwen3VL::Vision::Input> {
            static void write(Return<void>::Writer &writer, const Qwen3VL::Vision::Input &input) {
                writer.write_field(input.patches);
                writer.write_field(input.position_indices);
                writer.write_field(input.position_weights);
                writer.write_field(input.cos);
                writer.write_field(input.sin);
                writer.write_field(input.segments);
                writer.write_field(input.grids);
            }
    };

    template <> struct Argument<Qwen3VL::Vision::Input> {
            static Qwen3VL::Vision::Input read(Argument<void>::Reader &reader) {
                Qwen3VL::Vision::Input input;
                input.patches = reader.read_field<std::vector<std::uint16_t>>();
                input.position_indices = reader.read_field<std::vector<float>>();
                input.position_weights = reader.read_field<std::vector<float>>();
                input.cos = reader.read_field<std::vector<float>>();
                input.sin = reader.read_field<std::vector<float>>();
                input.segments = reader.read_field<std::vector<std::int32_t>>();
                input.grids = reader.read_field<std::vector<Qwen3VL::Vision::Grid>>();
                return input;
            }
    };
}
