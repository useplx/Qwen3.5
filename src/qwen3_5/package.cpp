#include "qwen3_5/nn/decoder.hpp"
#include "qwen3_5/nn/vision.hpp"

#include <syn/package.hpp>

PUBLISH(decoder, Qwen35::Nn::Decoder{})
PUBLISH(prefill, (Qwen35::Nn::Decoder{Qwen35::Config{}, Qwen35::Nn::Decoder::Output::State}))
PUBLISH(vision, Qwen35::Nn::Vision::Encoder{Qwen35::Config{}})
