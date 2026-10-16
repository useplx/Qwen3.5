#pragma once

#include "qwen3_5/config.hpp"

#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

#include <cstdint>
#include <string>
#include <utility>

namespace Qwen35::Nn::Vision {
    inline constexpr std::int64_t PatchElements = 3 * 2 * 16 * 16;
    inline constexpr std::int64_t HeadDim = 64;

    class Dense : public Mln::Sdk::Module {
        public:
            Dense(std::int64_t inputs, std::int64_t outputs);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &input);

        private:
            std::int64_t inputs_;
            std::int64_t outputs_;
    };

    class LayerNorm : public Mln::Sdk::Module {
        public:
            explicit LayerNorm(std::int64_t width);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &input);

        private:
            std::int64_t width_;
    };

    class Attention : public Mln::Sdk::Module {
        public:
            Attention();
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &hidden, const Mln::Sdk::Tensor &cos,
                                     const Mln::Sdk::Tensor &sin, const Mln::Sdk::Tensor &segments);
            std::vector<Mln::Sdk::Tensor> stages(const Mln::Sdk::Tensor &hidden,
                                                 const Mln::Sdk::Tensor &cos,
                                                 const Mln::Sdk::Tensor &sin,
                                                 const Mln::Sdk::Tensor &segments);
    };

    class Block : public Mln::Sdk::Module {
        public:
            explicit Block(std::int64_t layer);
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &hidden, const Mln::Sdk::Tensor &cos,
                                     const Mln::Sdk::Tensor &sin, const Mln::Sdk::Tensor &segments);
            std::pair<Mln::Sdk::Tensor, Mln::Sdk::Tensor> stages(const Mln::Sdk::Tensor &hidden,
                                                                 const Mln::Sdk::Tensor &cos,
                                                                 const Mln::Sdk::Tensor &sin,
                                                                 const Mln::Sdk::Tensor &segments);
            std::vector<Mln::Sdk::Tensor> attention_stages(const Mln::Sdk::Tensor &hidden,
                                                           const Mln::Sdk::Tensor &cos,
                                                           const Mln::Sdk::Tensor &sin,
                                                           const Mln::Sdk::Tensor &segments);
    };

    class Merger : public Mln::Sdk::Module {
        public:
            Merger();
            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &hidden);
    };

    class Encoder final : public Mln::Sdk::Module {
        public:
            explicit Encoder(Config config = {});

            std::vector<Mln::Sdk::Tensor> graph(std::span<const Mln::Sdk::Tensor> inputs) override;
            std::vector<std::vector<Mln::Sdk::Tensor>> specializations() const override;

            Mln::Sdk::Tensor forward(const Mln::Sdk::Tensor &patches,
                                     const Mln::Sdk::Tensor &position_indices,
                                     const Mln::Sdk::Tensor &position_weights,
                                     const Mln::Sdk::Tensor &cos, const Mln::Sdk::Tensor &sin,
                                     const Mln::Sdk::Tensor &segments);

        private:
            std::vector<Mln::Sdk::Tensor>
            stages(const Mln::Sdk::Tensor &patches, const Mln::Sdk::Tensor &position_indices,
                   const Mln::Sdk::Tensor &position_weights, const Mln::Sdk::Tensor &cos,
                   const Mln::Sdk::Tensor &sin, const Mln::Sdk::Tensor &segments);

            Config config_;
    };
}
