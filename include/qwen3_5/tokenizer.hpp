#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Qwen35 {
    struct Message {
            std::string role;
            std::string content;
    };

    class Tokenizer {
        public:
            explicit Tokenizer(std::span<const uint8_t> tokenizer_json);
            ~Tokenizer();

            Tokenizer(Tokenizer &&) noexcept;
            Tokenizer &operator=(Tokenizer &&) noexcept;
            Tokenizer(const Tokenizer &) = delete;
            Tokenizer &operator=(const Tokenizer &) = delete;

            std::vector<int64_t> encode(std::string_view text) const;
            std::string decode(std::span<const int64_t> ids, bool skip_special_tokens = true) const;

        private:
            struct Impl;
            std::unique_ptr<Impl> impl_;
    };

    std::string apply_chat_template(std::span<const Message> messages,
                                    bool add_generation_prompt = true);
}
