#include <mln/sdk/native.hpp>

#include "qwen3_vl/config.hpp"
#include "qwen3_vl/errors.hpp"
#include "qwen3_vl/tokenizer.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace {
    std::mutex tokenizer_mutex;
    std::unique_ptr<Qwen3VL::Tokenizer> cached_tokenizer;
    uint64_t cached_fingerprint = 0;
    const uint8_t *cached_data = nullptr;
    std::size_t cached_size = 0;

    uint64_t fingerprint(std::span<const uint8_t> bytes) {
        uint64_t value = 0xcbf29ce484222325ULL;
        for (uint8_t byte : bytes) {
            value ^= byte;
            value *= 0x100000001b3ULL;
        }
        return value;
    }
}

PUBLISH(qwen3vl, tokenizer, std::vector<uint8_t>, uint32_t mode, std::span<const uint8_t> data,
        std::span<const uint8_t> tokenizer_json) {
    std::lock_guard lock(tokenizer_mutex);
    const bool same_mapping = cached_tokenizer && tokenizer_json.data() == cached_data &&
                              tokenizer_json.size() == cached_size;
    const uint64_t current_fingerprint =
        same_mapping ? cached_fingerprint : fingerprint(tokenizer_json);
    if (!cached_tokenizer || cached_fingerprint != current_fingerprint) {
        cached_tokenizer = std::make_unique<Qwen3VL::Tokenizer>(tokenizer_json);
        cached_fingerprint = current_fingerprint;
    }
    cached_data = tokenizer_json.data();
    cached_size = tokenizer_json.size();

    if (mode == 0) {
        const std::string prompt(reinterpret_cast<const char *>(data.data()), data.size());
        const std::string chat =
            Qwen3VL::apply_chat_template(std::array<Qwen3VL::Message, 1>{{{"user", prompt}}});
        const std::vector<int64_t> ids = cached_tokenizer->encode(chat);
        std::vector<uint8_t> output(ids.size() * sizeof(float));
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const float token = static_cast<float>(ids[i]);
            std::memcpy(output.data() + i * sizeof(float), &token, sizeof(float));
        }
        return output;
    }
    if (mode == 2) {
        const std::string_view payload(reinterpret_cast<const char *>(data.data()), data.size());
        const std::size_t separator = payload.find('\0');
        if (separator == std::string_view::npos)
            Mln::raise(Qwen3VL::Errors::Tokenizer::Native::Chat::Separator::Missing);
        const std::array<Qwen3VL::Message, 2> messages{{
            {"system", std::string(payload.substr(0, separator))},
            {"user", std::string(payload.substr(separator + 1))},
        }};
        const std::string chat = Qwen3VL::apply_chat_template(messages);
        const std::vector<int64_t> ids = cached_tokenizer->encode(chat);
        std::vector<uint8_t> output(ids.size() * sizeof(float));
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const float token = static_cast<float>(ids[i]);
            std::memcpy(output.data() + i * sizeof(float), &token, sizeof(float));
        }
        return output;
    }
    if (mode == 3) {
        const std::string content(reinterpret_cast<const char *>(data.data()), data.size());
        const std::string chat =
            Qwen3VL::apply_chat_template(std::array<Qwen3VL::Message, 1>{{{"user", content}}});
        const std::vector<int64_t> ids = cached_tokenizer->encode(chat);
        std::vector<uint8_t> output(ids.size() * sizeof(float));
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const float token = static_cast<float>(ids[i]);
            std::memcpy(output.data() + i * sizeof(float), &token, sizeof(float));
        }
        return output;
    }
    if (mode == 1) {
        if (data.size() % sizeof(float) != 0)
            Mln::raise(Qwen3VL::Errors::Tokenizer::Native::Token::Storage::Misaligned, data.size());
        std::vector<int64_t> ids(data.size() / sizeof(float));
        for (std::size_t i = 0; i < ids.size(); ++i) {
            float token;
            std::memcpy(&token, data.data() + i * sizeof(float), sizeof(float));
            ids[i] = static_cast<int64_t>(token);
        }
        const std::string text = cached_tokenizer->decode(ids);
        return std::vector<uint8_t>(text.begin(), text.end());
    }
    Mln::raise(Qwen3VL::Errors::Tokenizer::Native::Mode::Unknown, mode);
}
