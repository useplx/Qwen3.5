#pragma once

#include <mln/error.hpp>

namespace Qwen35::Errors {
    namespace Config::Context {
        inline constexpr Mln::Error Invalid{4137, "Qwen3.5 received an invalid context size: {}"};
    }

    namespace Config::Generated::Token::Count {
        inline constexpr Mln::Error Invalid{
            4138, "Qwen3.5 received an invalid generated-token limit: {}"};
    }

    namespace Config::Prefill::Chunk {
        inline constexpr Mln::Error Invalid{
            4139, "Qwen3.5 received invalid prefill_chunk {}; expected a power of two"};
    }

    namespace Config::Vision::Patches {
        inline constexpr Mln::Error Invalid{
            4143, "Qwen3.5 received invalid vision_max_patches {}; expected a power of two >= 256"};
        inline constexpr Mln::Error Exceeded{
            4144, "Qwen3.5 produced {} visual patches; the profile supports at most {}"};
    }

    namespace Input::Magic {
        inline constexpr Mln::Error Invalid{4102, "Qwen3.5 input has invalid magic: {}"};
    }

    namespace Input::Content::Kind {
        inline constexpr Mln::Error Invalid{4140, "Qwen3.5 input has invalid kind {} at item {}"};
    }

    namespace Decoder::Graph::Input::Count {
        inline constexpr Mln::Error Invalid{4100,
                                            "Qwen3.5 decoder expected 10 inputs, received {}"};
    }

    namespace Decoder::Runtime::Output::Count {
        inline constexpr Mln::Error Invalid{4101,
                                            "Qwen3.5 decoder expected 3 outputs, received {}"};
    }

    namespace Decoder::Prefill::Runtime::Output::Count {
        inline constexpr Mln::Error Invalid{4142,
                                            "Qwen3.5 prefill expected 2 outputs, received {}"};
    }

    namespace Decoder::Prompt::Token::Count {
        inline constexpr Mln::Error Invalid{
            4131, "Qwen3.5 received a {}-token prompt for a context size of {}"};
    }

    namespace Decoder::Sampling::TopK {
        inline constexpr Mln::Error Invalid{4132, "Qwen3.5 received invalid top_k: {}"};
    }

    namespace Decoder::Sampling::TopP {
        inline constexpr Mln::Error Invalid{4133, "Qwen3.5 received invalid top_p: {}"};
    }

    namespace Decoder::Sampling::Temperature {
        inline constexpr Mln::Error Invalid{4134, "Qwen3.5 received invalid temperature: {}"};
    }

    namespace Decoder::Sampling::Repetition::Penalty {
        inline constexpr Mln::Error Invalid{4135,
                                            "Qwen3.5 received invalid repetition_penalty: {}"};
    }

    namespace Decoder::Sampling::Presence::Penalty {
        inline constexpr Mln::Error Invalid{4145, "Qwen3.5 received invalid presence_penalty: {}"};
    }

    namespace Vision::Graph::Input::Count {
        inline constexpr Mln::Error Invalid{4103, "Qwen3.5 vision expected 6 inputs, received {}"};
    }

    namespace Vision::Runtime::Output::Count {
        inline constexpr Mln::Error Invalid{4108, "Qwen3.5 vision expected 1 output, received {}"};
    }

    namespace Vision::Token::Count {
        inline constexpr Mln::Error Invalid{
            4104, "Qwen3.5 received {} image tokens, but the encoder produced {} embeddings"};
    }

    namespace Vision::Image::Size::Too {
        inline constexpr Mln::Error Large{
            4106, "Qwen3.5 received a {}-byte encoded image; the OpenCV limit is {}"};
    }

    namespace Vision::Image::Decode {
        inline constexpr Mln::Error Failed{4107, "Qwen3.5 could not decode the image"};
    }

    namespace Vision::Image::Aspect::Ratio {
        inline constexpr Mln::Error Unsupported{
            4127, "Qwen3.5 received an image with aspect ratio {}; the official limit is 200"};
    }

    namespace Vision::Video::Frame::Count {
        inline constexpr Mln::Error Invalid{4128, "Qwen3.5 received a video with {} frames"};
    }

    namespace Vision::Video::Frames::Per::Second {
        inline constexpr Mln::Error Invalid{4136, "Qwen3.5 received an invalid video FPS: {}"};
    }

    namespace Vision::Grid {
        inline constexpr Mln::Error Missing{4129, "Qwen3.5 found no metadata for visual item {}"};
        inline constexpr Mln::Error Unused{4130, "Qwen3.5 received {} visual grids without tokens"};
    }

    namespace Vision::Grid::Content::Mismatch {
        inline constexpr Mln::Error Invalid{
            4141, "Qwen3.5 received {} visual items, but the provider produced {} grids"};
    }

    namespace Tokenizer::Utf8::Start {
        inline constexpr Mln::Error Truncated{4110,
                                              "Tokenizer received truncated UTF-8 at the start"};
        inline constexpr Mln::Error Invalid{4111, "Tokenizer received an invalid UTF-8 prefix: {}"};
    }
    namespace Tokenizer::Utf8::Continuation {
        inline constexpr Mln::Error Truncated{4112,
                                              "Tokenizer received a truncated UTF-8 continuation"};
        inline constexpr Mln::Error Invalid{4113,
                                            "Tokenizer received an invalid continuation byte: {}"};
    }
    namespace Tokenizer::Utf8::Normalize {
        inline constexpr Mln::Error Failed{4114, "Tokenizer could not normalize UTF-8: {}"};
    }
    namespace Tokenizer::Merge {
        inline constexpr Mln::Error Invalid{4115, "Tokenizer found an invalid BPE merge: {}"};
    }
    namespace Tokenizer::Regex::Compile {
        inline constexpr Mln::Error Failed{
            4116, "Tokenizer could not compile the official regex at offset {}"};
    }
    namespace Tokenizer::Symbol {
        inline constexpr Mln::Error Missing{4117, "Tokenizer did not find BPE symbol: {}"};
    }
    namespace Tokenizer::PreTokenize::Match {
        inline constexpr Mln::Error Missing{4118, "Tokenizer did not match input at offset {}"};
    }
    namespace Tokenizer::PreTokenize::Range {
        inline constexpr Mln::Error Invalid{4119, "Tokenizer found a gap at offset {}"};
    }
    namespace Tokenizer::Token::Id::Out::Of {
        inline constexpr Mln::Error Range{4120, "Tokenizer received invalid token id: {}"};
    }
    namespace Tokenizer::Byte::Codepoint {
        inline constexpr Mln::Error Unknown{4121,
                                            "Tokenizer did not recognize byte-level codepoint: {}"};
    }
    namespace Tokenizer::Chat::Role {
        inline constexpr Mln::Error Unsupported{4122, "Chat received unsupported role: {}"};
    }
    namespace Tokenizer::Chat::System::Position {
        inline constexpr Mln::Error Invalid{4123, "The system message must be first in the chat"};
    }
    namespace Tokenizer::Native::Chat::Separator {
        inline constexpr Mln::Error Missing{
            4124, "Native Tokenizer requires separate system and user messages"};
    }
    namespace Tokenizer::Native::Token::Storage {
        inline constexpr Mln::Error Misaligned{4125,
                                               "Native Tokenizer received {} bytes of tokens"};
    }
    namespace Tokenizer::Native::Mode {
        inline constexpr Mln::Error Unknown{4126, "Native Tokenizer received unknown mode: {}"};
    }
}
