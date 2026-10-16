#pragma once

#include "qwen3_5/model.hpp"

namespace Qwen35 {
    class Orchestration : public Model {
        public:
            Mln::Buffer forward(Mln::Runtime::Session &session, Mln::Buffer input) override;
    };
}
