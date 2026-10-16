#pragma once

#include "qwen3_vl/model.hpp"

namespace Qwen3VL {
    class Orchestration : public Model {
        public:
            Mln::Buffer forward(Mln::Runtime::Session &session, Mln::Buffer input) override;
    };
}
