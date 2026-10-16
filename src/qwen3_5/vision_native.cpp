#include "qwen3_5/config.hpp"
#include "qwen3_5/errors.hpp"
#include "qwen3_5/float16.hpp"
#include "qwen3_5/vision.hpp"

#include <mln/sdk/native.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
    using namespace Qwen35::Arch;

    constexpr int Factor = VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE;
    constexpr std::int64_t ImageMinPixels = 65'536;
    constexpr std::int64_t DefaultImageMaxPixels = 16'777'216;

    std::int64_t image_max_pixels() {
        const char *configured = std::getenv("SENSIX_IMAGE_MAX_PIXELS");
        if (!configured || !*configured)
            return DefaultImageMaxPixels;
        char *end = nullptr;
        const long long value = std::strtoll(configured, &end, 10);
        return end && *end == '\0' && value >= ImageMinPixels ? value : DefaultImageMaxPixels;
    }

    int rounded_multiple(int value) {
        const int quotient = value / Factor;
        const int remainder = value % Factor;
        if (remainder * 2 < Factor)
            return quotient * Factor;
        if (remainder * 2 > Factor)
            return (quotient + 1) * Factor;
        return (quotient + quotient % 2) * Factor;
    }

    cv::Mat decode(std::span<const std::uint8_t> encoded) {
        if (encoded.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            Mln::raise(Qwen35::Errors::Vision::Image::Size::Too::Large, encoded.size(),
                       std::numeric_limits<int>::max());
        cv::Mat bytes(1, static_cast<int>(encoded.size()), CV_8UC1,
                      const_cast<std::uint8_t *>(encoded.data()));
        cv::Mat image = cv::imdecode(bytes, cv::IMREAD_COLOR);
        if (image.empty())
            Mln::raise(Qwen35::Errors::Vision::Image::Decode::Failed);
        const char *configured = std::getenv("SENSIX_CAMERA_ROTATION");
        const int rotation = configured && *configured ? std::atoi(configured) : 0;
        if (rotation == 90)
            cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
        else if (rotation == 180)
            cv::rotate(image, image, cv::ROTATE_180);
        else if (rotation == 270)
            cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
        else if (rotation != 0)
            Mln::raise(Qwen35::Errors::Vision::Image::Decode::Failed);
        return image;
    }

    cv::Size smart_size(const cv::Mat &image) {
        const double ratio = static_cast<double>(std::max(image.rows, image.cols)) /
                             static_cast<double>(std::min(image.rows, image.cols));
        if (ratio > 200.0)
            Mln::raise(Qwen35::Errors::Vision::Image::Aspect::Ratio::Unsupported, ratio);

        int height = rounded_multiple(image.rows);
        int width = rounded_multiple(image.cols);
        const std::int64_t pixels = static_cast<std::int64_t>(height) * width;
        const std::int64_t max_pixels = image_max_pixels();
        if (pixels > max_pixels) {
            const double scale =
                std::sqrt(static_cast<double>(image.rows) * image.cols / max_pixels);
            height = std::max(Factor,
                              static_cast<int>(std::floor(image.rows / scale / Factor)) * Factor);
            width = std::max(Factor,
                             static_cast<int>(std::floor(image.cols / scale / Factor)) * Factor);
        } else if (pixels < ImageMinPixels) {
            const double scale =
                std::sqrt(static_cast<double>(ImageMinPixels) / (image.rows * image.cols));
            height = static_cast<int>(std::ceil(image.rows * scale / Factor)) * Factor;
            width = static_cast<int>(std::ceil(image.cols * scale / Factor)) * Factor;
        }
        return {width, height};
    }

    cv::Size smart_video_size(const cv::Mat &frame, std::size_t frames) {
        const double ratio = static_cast<double>(std::max(frame.rows, frame.cols)) /
                             static_cast<double>(std::min(frame.rows, frame.cols));
        if (ratio > 200.0)
            Mln::raise(Qwen35::Errors::Vision::Image::Aspect::Ratio::Unsupported, ratio);

        const std::int64_t temporal =
            static_cast<std::int64_t>((frames + VISION_TEMPORAL_PATCH_SIZE - 1) /
                                      VISION_TEMPORAL_PATCH_SIZE) *
            VISION_TEMPORAL_PATCH_SIZE;
        int height = rounded_multiple(frame.rows);
        int width = rounded_multiple(frame.cols);
        const std::int64_t pixels = temporal * height * width;
        const std::int64_t max_pixels = image_max_pixels();
        if (pixels > max_pixels) {
            const double scale =
                std::sqrt(static_cast<double>(frames) * frame.rows * frame.cols / max_pixels);
            height = std::max(Factor,
                              static_cast<int>(std::floor(frame.rows / scale / Factor)) * Factor);
            width = std::max(Factor,
                             static_cast<int>(std::floor(frame.cols / scale / Factor)) * Factor);
        } else if (pixels < ImageMinPixels) {
            const double scale = std::sqrt(static_cast<double>(ImageMinPixels) /
                                           (static_cast<double>(frames) * frame.rows * frame.cols));
            height = static_cast<int>(std::ceil(frame.rows * scale / Factor)) * Factor;
            width = static_cast<int>(std::ceil(frame.cols * scale / Factor)) * Factor;
        }
        return {width, height};
    }

    cv::Mat prepare(const cv::Mat &image, cv::Size size) {
        cv::Mat resized;
        cv::resize(image, resized, size, 0.0, 0.0, cv::INTER_CUBIC);
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        return resized;
    }

    void append_positions(Qwen35::Vision::Input &output, int grid_height, int grid_width,
                          std::int32_t segment) {
        constexpr int EmbeddingSide = 48;
        for (int block_y = 0; block_y < grid_height / VISION_SPATIAL_MERGE_SIZE; ++block_y) {
            for (int block_x = 0; block_x < grid_width / VISION_SPATIAL_MERGE_SIZE; ++block_x) {
                for (int merge_y = 0; merge_y < VISION_SPATIAL_MERGE_SIZE; ++merge_y) {
                    for (int merge_x = 0; merge_x < VISION_SPATIAL_MERGE_SIZE; ++merge_x) {
                        const int row = block_y * VISION_SPATIAL_MERGE_SIZE + merge_y;
                        const int column = block_x * VISION_SPATIAL_MERGE_SIZE + merge_x;
                        const float source_y = static_cast<float>(row) * (EmbeddingSide - 1) /
                                               static_cast<float>(grid_height - 1);
                        const float source_x = static_cast<float>(column) * (EmbeddingSide - 1) /
                                               static_cast<float>(grid_width - 1);
                        const int y0 = static_cast<int>(std::floor(source_y));
                        const int x0 = static_cast<int>(std::floor(source_x));
                        const int y1 = std::min(y0 + 1, EmbeddingSide - 1);
                        const int x1 = std::min(x0 + 1, EmbeddingSide - 1);
                        const float fy = source_y - y0;
                        const float fx = source_x - x0;
                        const std::array<int, 4> indices{
                            y0 * EmbeddingSide + x0, y0 * EmbeddingSide + x1,
                            y1 * EmbeddingSide + x0, y1 * EmbeddingSide + x1};
                        const std::array<float, 4> weights{
                            (1.0F - fy) * (1.0F - fx), (1.0F - fy) * fx, fy * (1.0F - fx), fy * fx};
                        for (std::size_t corner = 0; corner < indices.size(); ++corner) {
                            output.position_indices.push_back(static_cast<float>(indices[corner]));
                            output.position_weights.push_back(weights[corner]);
                        }

                        std::array<float, 16> row_cosines{};
                        std::array<float, 16> row_sines{};
                        std::array<float, 16> column_cosines{};
                        std::array<float, 16> column_sines{};
                        for (int frequency = 0; frequency < 16; ++frequency) {
                            const float inverse =
                                1.0F /
                                std::pow(10000.0F, static_cast<float>(frequency * 2) / 32.0F);
                            const float row_angle = static_cast<float>(row) * inverse;
                            const float column_angle = static_cast<float>(column) * inverse;
                            row_cosines[frequency] = std::cos(row_angle);
                            row_sines[frequency] = std::sin(row_angle);
                            column_cosines[frequency] = std::cos(column_angle);
                            column_sines[frequency] = std::sin(column_angle);
                        }
                        output.cos.insert(output.cos.end(), row_cosines.begin(), row_cosines.end());
                        output.cos.insert(output.cos.end(), column_cosines.begin(),
                                          column_cosines.end());
                        output.cos.insert(output.cos.end(), row_cosines.begin(), row_cosines.end());
                        output.cos.insert(output.cos.end(), column_cosines.begin(),
                                          column_cosines.end());
                        output.sin.insert(output.sin.end(), row_sines.begin(), row_sines.end());
                        output.sin.insert(output.sin.end(), column_sines.begin(),
                                          column_sines.end());
                        output.sin.insert(output.sin.end(), row_sines.begin(), row_sines.end());
                        output.sin.insert(output.sin.end(), column_sines.begin(),
                                          column_sines.end());
                        output.segments.push_back(segment);
                    }
                }
            }
        }
    }

    void append_patches(Qwen35::Vision::Input &output, const cv::Mat &first,
                        const cv::Mat &second) {
        const int grid_height = first.rows / VISION_PATCH_SIZE;
        const int grid_width = first.cols / VISION_PATCH_SIZE;
        for (int block_y = 0; block_y < grid_height / VISION_SPATIAL_MERGE_SIZE; ++block_y) {
            for (int block_x = 0; block_x < grid_width / VISION_SPATIAL_MERGE_SIZE; ++block_x) {
                for (int merge_y = 0; merge_y < VISION_SPATIAL_MERGE_SIZE; ++merge_y) {
                    for (int merge_x = 0; merge_x < VISION_SPATIAL_MERGE_SIZE; ++merge_x) {
                        const int patch_y = block_y * VISION_SPATIAL_MERGE_SIZE + merge_y;
                        const int patch_x = block_x * VISION_SPATIAL_MERGE_SIZE + merge_x;
                        for (int channel = 0; channel < VISION_IN_CHANNELS; ++channel) {
                            for (const cv::Mat *frame : {&first, &second}) {
                                for (int y = 0; y < VISION_PATCH_SIZE; ++y) {
                                    const cv::Vec3b *row =
                                        frame->ptr<cv::Vec3b>(patch_y * VISION_PATCH_SIZE + y);
                                    for (int x = 0; x < VISION_PATCH_SIZE; ++x) {
                                        const float value =
                                            static_cast<float>(
                                                row[patch_x * VISION_PATCH_SIZE + x][channel]) /
                                                127.5F -
                                            1.0F;
                                        output.patches.push_back(Qwen35::to_float16(value));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void append_image(Qwen35::Vision::Input &output, const Qwen35::Image &image,
                      std::int32_t &segment) {
        const cv::Mat decoded = decode(image.encoded);
        const cv::Mat frame = prepare(decoded, smart_size(decoded));
        const int grid_height = frame.rows / VISION_PATCH_SIZE;
        const int grid_width = frame.cols / VISION_PATCH_SIZE;
        append_patches(output, frame, frame);
        append_positions(output, grid_height, grid_width, segment++);
        output.grids.push_back({Qwen35::Vision::Kind::Image, 1,
                                static_cast<std::uint32_t>(grid_height),
                                static_cast<std::uint32_t>(grid_width), 0.0F});
    }

    void append_video(Qwen35::Vision::Input &output, const Qwen35::Video &video,
                      std::int32_t &segment) {
        if (video.frames.empty())
            Mln::raise(Qwen35::Errors::Vision::Video::Frame::Count::Invalid, 0);
        if (!(video.frames_per_second > 0.0F) || !std::isfinite(video.frames_per_second))
            Mln::raise(Qwen35::Errors::Vision::Video::Frames::Per::Second::Invalid,
                       video.frames_per_second);
        const cv::Mat first_decoded = decode(video.frames.front().encoded);
        const cv::Size size = smart_video_size(first_decoded, video.frames.size());
        std::vector<cv::Mat> frames;
        frames.reserve(video.frames.size() + 1);
        frames.push_back(prepare(first_decoded, size));
        for (std::size_t index = 1; index < video.frames.size(); ++index)
            frames.push_back(prepare(decode(video.frames[index].encoded), size));
        if (frames.size() % 2 != 0)
            frames.push_back(frames.back());

        const int grid_height = size.height / VISION_PATCH_SIZE;
        const int grid_width = size.width / VISION_PATCH_SIZE;
        for (std::size_t index = 0; index < frames.size(); index += 2) {
            append_patches(output, frames[index], frames[index + 1]);
            append_positions(output, grid_height, grid_width, segment++);
        }
        const std::uint32_t temporal = static_cast<std::uint32_t>(frames.size() / 2);
        const float seconds =
            static_cast<float>(VISION_TEMPORAL_PATCH_SIZE) / video.frames_per_second;
        output.grids.push_back({Qwen35::Vision::Kind::Video, temporal,
                                static_cast<std::uint32_t>(grid_height),
                                static_cast<std::uint32_t>(grid_width), seconds});
    }
} // namespace

PUBLISH(qwen35, vision, Qwen35::Vision::Input, Qwen35::Vision::Request request) {
    Qwen35::Vision::Input output;
    std::int32_t segment = 0;
    for (const Qwen35::Vision::Request::Media &media : request.media) {
        std::visit(
            [&](const auto &value) {
                using T = std::remove_cvref_t<decltype(value)>;
                if constexpr (std::is_same_v<T, Qwen35::Image>)
                    append_image(output, value, segment);
                else
                    append_video(output, value, segment);
            },
            media);
    }
    return output;
}
