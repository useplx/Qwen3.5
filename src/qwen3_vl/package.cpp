#include "qwen3_vl/nn/decoder.hpp"
#include "qwen3_vl/nn/vision.hpp"

#include <syn/package.hpp>

PUBLISH(decoder, Qwen3VL::Nn::Decoder{})
PUBLISH(prefill, (Qwen3VL::Nn::Decoder{Qwen3VL::Config{}, Qwen3VL::Nn::Decoder::Output::State}))
PUBLISH(vision, Qwen3VL::Nn::Vision::Encoder{Qwen3VL::Config{}})
