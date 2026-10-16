#include "qwen3_5/nn/rmsnorm.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/operations.hpp>

#include <cmath>

using namespace Mln::Sdk;

namespace Qwen35::Nn {
    namespace {
        Tensor scalar_broadcast(Operations &ops, double value, const std::vector<int64_t> &shape,
                                std::optional<Mln::DType> dtype = std::nullopt) {
            Tensor s = dtype ? ops.insert<Functional::Const>(Weight{"", {}, value}, false, *dtype)
                             : ops.insert<Functional::Const>(Weight{"", {}, value});
            return ops.insert<Functional::Broadcast>(std::move(s), shape, std::vector<int64_t>{});
        }

        std::vector<int64_t> leading_dims(int64_t rank) {
            std::vector<int64_t> dims(static_cast<std::size_t>(rank));
            for (int64_t i = 0; i < rank; i++)
                dims[i] = i;
            return dims;
        }
        Tensor rms_norm_impl(Operations &ops, const Tensor &x, const Tensor &weight, int64_t dim,
                             double eps, std::optional<Mln::DType> computation_dtype,
                             std::optional<Mln::DType> output_dtype, bool zero_centered) {
            const int64_t rank = static_cast<int64_t>(x.shape.size());
            std::vector<int64_t> variance_shape(x.shape.begin(), x.shape.end() - 1);
            Tensor computation = x;
            if (computation_dtype && computation.dtype != *computation_dtype)
                computation =
                    ops.insert<Functional::Convert>(std::move(computation), *computation_dtype);

            Tensor scaled = ops.insert<Functional::Multiply>(
                computation, scalar_broadcast(ops, 1.0 / std::sqrt(static_cast<double>(dim)),
                                              x.shape, computation_dtype));
            Tensor scaled_copy = scaled;
            Tensor squared =
                ops.insert<Functional::Multiply>(std::move(scaled), std::move(scaled_copy));
            Tensor variance = ops.insert<Functional::ReduceSum>(std::move(squared),
                                                                std::vector<int64_t>{rank - 1});
            variance = ops.insert<Functional::Add>(
                std::move(variance), scalar_broadcast(ops, eps, variance_shape, computation_dtype));
            Tensor inv = ops.insert<Functional::Rsqrt>(std::move(variance));
            Tensor broadcast_inverse =
                ops.insert<Functional::Broadcast>(std::move(inv), x.shape, leading_dims(rank - 1));
            Tensor normed = ops.insert<Functional::Multiply>(std::move(computation),
                                                             std::move(broadcast_inverse));
            if (output_dtype && output_dtype != computation_dtype)
                normed = ops.insert<Functional::Convert>(std::move(normed), *output_dtype);

            Tensor effective_weight = weight;
            if (output_dtype && output_dtype == computation_dtype)
                effective_weight =
                    ops.insert<Functional::Convert>(std::move(effective_weight), *output_dtype);
            if (zero_centered)
                effective_weight = ops.insert<Functional::Add>(
                    std::move(effective_weight), scalar_broadcast(ops, 1.0, {dim}, output_dtype));
            Tensor broadcast_weight = ops.insert<Functional::Broadcast>(
                std::move(effective_weight), x.shape, std::vector<int64_t>{rank - 1});
            Tensor result =
                ops.insert<Functional::Multiply>(std::move(normed), std::move(broadcast_weight));
            return result;
        }
    }

    Tensor rms_norm(Operations &ops, const Tensor &x, const Tensor &weight, int64_t dim, double eps,
                    bool zero_centered) {
        return rms_norm_impl(ops, x, weight, dim, eps, Mln::DType::Float32, Mln::DType::Float16,
                             zero_centered);
    }

    Tensor rms_norm(Operations &ops, const Tensor &x, const Tensor &weight, int64_t dim, double eps,
                    Mln::DType computation_dtype, bool zero_centered) {
        return rms_norm_impl(ops, x, weight, dim, eps, computation_dtype, computation_dtype,
                             zero_centered);
    }

    RMSNorm::RMSNorm(int64_t dim, double eps, bool zero_centered)
        : dim_(dim), eps_(eps), zero_centered_(zero_centered) {
        parameters.insert("weight", {{}, {dim}});
    }

    Tensor RMSNorm::forward(const Tensor &x) {
        return rms_norm(operations, x, parameters.at("weight"), dim_, eps_, zero_centered_);
    }
}
