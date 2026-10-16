#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <utf8proc.h>

#include "qwen3_5/errors.hpp"
#include "qwen3_5/tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Qwen35 {
    namespace {
        using Json = nlohmann::json;

        std::string json_text(std::span<const uint8_t> bytes) {
            return std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        }

        std::string pair_key(std::string_view left, std::string_view right) {
            std::string key;
            key.reserve(left.size() + right.size() + 1);
            key.append(left);
            key.push_back('\0');
            key.append(right);
            return key;
        }

        void append_utf8(std::string &output, uint32_t codepoint) {
            if (codepoint <= 0x7f) {
                output.push_back(static_cast<char>(codepoint));
            } else if (codepoint <= 0x7ff) {
                output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
                output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
            } else if (codepoint <= 0xffff) {
                output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
                output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
                output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
            } else {
                output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
                output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
                output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
                output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
            }
        }

        uint32_t next_utf8(std::string_view text, std::size_t &offset) {
            if (offset >= text.size())
                Mln::raise(Errors::Tokenizer::Utf8::Start::Truncated);
            const uint8_t first = static_cast<uint8_t>(text[offset++]);
            if (first < 0x80)
                return first;
            int continuation = 0;
            uint32_t value = 0;
            if ((first & 0xe0) == 0xc0) {
                continuation = 1;
                value = first & 0x1f;
            } else if ((first & 0xf0) == 0xe0) {
                continuation = 2;
                value = first & 0x0f;
            } else if ((first & 0xf8) == 0xf0) {
                continuation = 3;
                value = first & 0x07;
            } else
                Mln::raise(Errors::Tokenizer::Utf8::Start::Invalid, first);
            for (int i = 0; i < continuation; ++i) {
                if (offset >= text.size())
                    Mln::raise(Errors::Tokenizer::Utf8::Continuation::Truncated);
                const uint8_t byte = static_cast<uint8_t>(text[offset++]);
                if ((byte & 0xc0) != 0x80)
                    Mln::raise(Errors::Tokenizer::Utf8::Continuation::Invalid, byte);
                value = (value << 6) | (byte & 0x3f);
            }
            return value;
        }

        std::string normalize_nfc(std::string_view input) {
            utf8proc_uint8_t *output = nullptr;
            const utf8proc_ssize_t size =
                utf8proc_map(reinterpret_cast<const utf8proc_uint8_t *>(input.data()),
                             static_cast<utf8proc_ssize_t>(input.size()), &output,
                             static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
            if (size < 0)
                Mln::raise(Errors::Tokenizer::Utf8::Normalize::Failed, utf8proc_errmsg(size));
            const std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> owned(output, &std::free);
            std::string normalized(reinterpret_cast<const char *>(owned.get()),
                                   static_cast<std::size_t>(size));
            return normalized;
        }
    }

    namespace Regex {
        struct Deleter {
                void operator()(pcre2_code *value) const { pcre2_code_free(value); }
        };
    }

    namespace Match {
        struct Deleter {
                void operator()(pcre2_match_data *value) const { pcre2_match_data_free(value); }
        };
    }

    struct Tokenizer::Impl {
            std::unordered_map<std::string, int64_t> vocab;
            std::vector<std::string> id_to_token;
            std::unordered_map<std::string, int64_t> merge_rank;
            std::unordered_map<std::string, int64_t> special_ids;
            std::unordered_set<int64_t> special_id_set;
            std::vector<std::pair<std::string, int64_t>> specials;
            std::array<std::string, 256> byte_token;
            std::unordered_map<uint32_t, uint8_t> token_byte;
            std::unique_ptr<pcre2_code, Regex::Deleter> regex;
            mutable std::unordered_map<std::string, std::vector<int64_t>> cache;

            explicit Impl(std::span<const uint8_t> bytes) {
                const Json root = Json::parse(json_text(bytes));
                const Json &model = root.at("model");
                int64_t maximum_id = 0;
                for (const auto &[token, value] : model.at("vocab").items()) {
                    const int64_t id = value.get<int64_t>();
                    vocab.emplace(token, id);
                    maximum_id = std::max(maximum_id, id);
                }
                for (const Json &item : root.at("added_tokens")) {
                    const std::string token = item.at("content").get<std::string>();
                    const int64_t id = item.at("id").get<int64_t>();
                    vocab.emplace(token, id);
                    special_ids.emplace(token, id);
                    special_id_set.insert(id);
                    specials.emplace_back(token, id);
                    maximum_id = std::max(maximum_id, id);
                }
                std::sort(specials.begin(), specials.end(), [](const auto &a, const auto &b) {
                    return a.first.size() > b.first.size();
                });
                id_to_token.resize(static_cast<std::size_t>(maximum_id + 1));
                for (const auto &[token, id] : vocab)
                    id_to_token.at(static_cast<std::size_t>(id)) = token;

                int64_t rank = 0;
                for (const Json &merge : model.at("merges")) {
                    std::string left;
                    std::string right;
                    if (merge.is_array()) {
                        left = merge.at(0).get<std::string>();
                        right = merge.at(1).get<std::string>();
                    } else {
                        const std::string joined = merge.get<std::string>();
                        const std::size_t separator = joined.find(' ');
                        if (separator == std::string::npos)
                            Mln::raise(Errors::Tokenizer::Merge::Invalid, joined);
                        left = joined.substr(0, separator);
                        right = joined.substr(separator + 1);
                    }
                    merge_rank.emplace(pair_key(left, right), rank++);
                }

                std::vector<uint32_t> codepoints;
                for (uint32_t c = 33; c <= 126; ++c)
                    codepoints.push_back(c);
                for (uint32_t c = 161; c <= 172; ++c)
                    codepoints.push_back(c);
                for (uint32_t c = 174; c <= 255; ++c)
                    codepoints.push_back(c);
                std::array<bool, 256> direct{};
                for (uint32_t c : codepoints)
                    direct[c] = true;
                uint32_t extra = 0;
                for (uint32_t byte = 0; byte < 256; ++byte) {
                    const uint32_t mapped = direct[byte] ? byte : 256 + extra++;
                    append_utf8(byte_token[byte], mapped);
                    token_byte.emplace(mapped, static_cast<uint8_t>(byte));
                }

                static constexpr char pattern[] =
                    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
                    "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
                    "\\s+(?!\\S)|\\s+";
                int error = 0;
                PCRE2_SIZE error_offset = 0;
                regex.reset(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern),
                                          PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP, &error,
                                          &error_offset, nullptr));
                if (!regex)
                    Mln::raise(Errors::Tokenizer::Regex::Compile::Failed, error_offset);
            }

            std::vector<int64_t> bpe(std::string word) const {
                if (const auto found = cache.find(word); found != cache.end())
                    return found->second;

                std::vector<std::string> pieces;
                for (std::size_t offset = 0; offset < word.size();) {
                    const uint8_t byte = static_cast<uint8_t>(word[offset++]);
                    pieces.push_back(byte_token[byte]);
                }
                while (pieces.size() > 1) {
                    int64_t best_rank = std::numeric_limits<int64_t>::max();
                    std::string best_left;
                    std::string best_right;
                    for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
                        const auto found = merge_rank.find(pair_key(pieces[i], pieces[i + 1]));
                        if (found != merge_rank.end() && found->second < best_rank) {
                            best_rank = found->second;
                            best_left = pieces[i];
                            best_right = pieces[i + 1];
                        }
                    }
                    if (best_rank == std::numeric_limits<int64_t>::max())
                        break;
                    std::vector<std::string> merged;
                    merged.reserve(pieces.size());
                    for (std::size_t i = 0; i < pieces.size();) {
                        if (i + 1 < pieces.size() && pieces[i] == best_left &&
                            pieces[i + 1] == best_right) {
                            merged.push_back(pieces[i] + pieces[i + 1]);
                            i += 2;
                        } else {
                            merged.push_back(std::move(pieces[i++]));
                        }
                    }
                    pieces = std::move(merged);
                }

                std::vector<int64_t> ids;
                ids.reserve(pieces.size());
                for (const std::string &piece : pieces) {
                    const auto found = vocab.find(piece);
                    if (found == vocab.end())
                        Mln::raise(Errors::Tokenizer::Symbol::Missing, piece);
                    ids.push_back(found->second);
                }
                cache.emplace(std::move(word), ids);
                return ids;
            }

            void encode_normal(std::string_view text, std::vector<int64_t> &output) const {
                if (text.empty())
                    return;
                std::unique_ptr<pcre2_match_data, Match::Deleter> match(
                    pcre2_match_data_create_from_pattern(regex.get(), nullptr));
                std::size_t offset = 0;
                while (offset < text.size()) {
                    const int result =
                        pcre2_match(regex.get(), reinterpret_cast<PCRE2_SPTR>(text.data()),
                                    text.size(), offset, 0, match.get(), nullptr);
                    if (result < 0)
                        Mln::raise(Errors::Tokenizer::PreTokenize::Match::Missing, offset);
                    PCRE2_SIZE *range = pcre2_get_ovector_pointer(match.get());
                    if (range[0] != offset || range[1] <= range[0])
                        Mln::raise(Errors::Tokenizer::PreTokenize::Range::Invalid, offset);
                    std::vector<int64_t> ids = bpe(
                        std::string(text.substr(static_cast<std::size_t>(range[0]),
                                                static_cast<std::size_t>(range[1] - range[0]))));
                    output.insert(output.end(), ids.begin(), ids.end());
                    offset = static_cast<std::size_t>(range[1]);
                }
            }
    };

    Tokenizer::Tokenizer(std::span<const uint8_t> tokenizer_json)
        : impl_(std::make_unique<Impl>(tokenizer_json)) {}
    Tokenizer::~Tokenizer() = default;
    Tokenizer::Tokenizer(Tokenizer &&) noexcept = default;
    Tokenizer &Tokenizer::operator=(Tokenizer &&) noexcept = default;

    std::vector<int64_t> Tokenizer::encode(std::string_view text) const {
        const std::string normalized = normalize_nfc(text);
        std::vector<int64_t> output;
        std::size_t cursor = 0;
        while (cursor < normalized.size()) {
            std::size_t special_at = std::string::npos;
            const std::pair<std::string, int64_t> *selected = nullptr;
            for (const auto &special : impl_->specials) {
                const std::size_t found = normalized.find(special.first, cursor);
                if (found < special_at) {
                    special_at = found;
                    selected = &special;
                }
            }
            if (!selected) {
                impl_->encode_normal(std::string_view(normalized).substr(cursor), output);
                break;
            }
            impl_->encode_normal(std::string_view(normalized).substr(cursor, special_at - cursor),
                                 output);
            output.push_back(selected->second);
            cursor = special_at + selected->first.size();
        }
        return output;
    }

    std::string Tokenizer::decode(std::span<const int64_t> ids, bool skip_special_tokens) const {
        std::string encoded;
        for (int64_t id : ids) {
            if (id < 0 || static_cast<std::size_t>(id) >= impl_->id_to_token.size() ||
                impl_->id_to_token[static_cast<std::size_t>(id)].empty())
                Mln::raise(Errors::Tokenizer::Token::Id::Out::Of::Range, id);
            if (skip_special_tokens && impl_->special_id_set.contains(id))
                continue;
            encoded += impl_->id_to_token[static_cast<std::size_t>(id)];
        }
        std::string output;
        for (std::size_t offset = 0; offset < encoded.size();) {
            const uint32_t codepoint = next_utf8(encoded, offset);
            const auto found = impl_->token_byte.find(codepoint);
            if (found == impl_->token_byte.end())
                Mln::raise(Errors::Tokenizer::Byte::Codepoint::Unknown, codepoint);
            output.push_back(static_cast<char>(found->second));
        }
        return output;
    }

    std::string apply_chat_template(std::span<const Message> messages, bool add_generation_prompt) {
        const auto trim = [](std::string_view value) {
            const std::size_t begin = value.find_first_not_of(" \t\r\n");
            if (begin == std::string_view::npos)
                return std::string{};
            const std::size_t end = value.find_last_not_of(" \t\r\n");
            return std::string(value.substr(begin, end - begin + 1));
        };
        std::string result;
        for (std::size_t index = 0; index < messages.size(); ++index) {
            const Message &message = messages[index];
            if (message.role != "system" && message.role != "user" && message.role != "assistant")
                Mln::raise(Errors::Tokenizer::Chat::Role::Unsupported, message.role);
            if (message.role == "system" && index != 0)
                Mln::raise(Errors::Tokenizer::Chat::System::Position::Invalid);
            result += "<|im_start|>";
            result += message.role;
            result.push_back('\n');
            result += trim(message.content);
            result += "<|im_end|>\n";
        }
        if (add_generation_prompt)
            result += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        return result;
    }
}
