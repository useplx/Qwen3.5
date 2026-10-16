#pragma once

#include "qwen3_5/config.hpp"

#include <mln/sdk/module.hpp>
#include <mln/sdk/tensor.hpp>

#include <string>
#include <tuple>

namespace Qwen35::Nn {
    class Decoder final : public Mln::Sdk::Module {
        public:
            class Layer;

            enum class Output {
                Logits,
                State,
            };

            explicit Decoder(Config config = {}, Output output = Output::Logits);

            std::tuple<Mln::Sdk::Tensor, Mln::Sdk::Tensor, Mln::Sdk::Tensor>
            forward(const Mln::Sdk::Tensor &token, const Mln::Sdk::Tensor &storage,
                    const Mln::Sdk::Tensor &recurrent, const Mln::Sdk::Tensor &cos,
                    const Mln::Sdk::Tensor &sin, const Mln::Sdk::Tensor &position,
                    const Mln::Sdk::Tensor &mask, const Mln::Sdk::Tensor &visual_embeddings,
                    const Mln::Sdk::Tensor &visual_indices, const Mln::Sdk::Tensor &visual_mask);

            std::vector<Mln::Sdk::Tensor> graph(std::span<const Mln::Sdk::Tensor> inputs) override;
            std::vector<std::vector<Mln::Sdk::Tensor>> specializations() const override;
            std::string entrypoint() const override;

            static int64_t storage_elements(const Config &config);
            static int64_t recurrent_elements();

        private:
            struct Transition {
                    Mln::Sdk::Tensor hidden;
                    Mln::Sdk::Tensor storage;
                    Mln::Sdk::Tensor recurrent;
            };

            Transition transition(const Mln::Sdk::Tensor &token, const Mln::Sdk::Tensor &storage,
                                  const Mln::Sdk::Tensor &recurrent, const Mln::Sdk::Tensor &cos,
                                  const Mln::Sdk::Tensor &sin, const Mln::Sdk::Tensor &position,
                                  const Mln::Sdk::Tensor &mask,
                                  const Mln::Sdk::Tensor &visual_embeddings,
                                  const Mln::Sdk::Tensor &visual_indices,
                                  const Mln::Sdk::Tensor &visual_mask);

            Config config_;
            Output output_;
    };
}
