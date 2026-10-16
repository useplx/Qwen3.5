#include "qwen3_vl/nn/attention.hpp"
#include "qwen3_vl/config.hpp"
#include "qwen3_vl/nn/linear.hpp"
#include "qwen3_vl/nn/rmsnorm.hpp"

#include <mln/sdk/functional.hpp>
#include <mln/sdk/operations.hpp>

#include <cmath>

using namespace Mln::Sdk;
using namespace Qwen3VL::Arch;

namespace Qwen3VL::Nn {
    namespace {
        constexpr int64_t Q_DIM = NUM_HEADS * HEAD_DIM;
        constexpr int64_t KV_DIM = NUM_KV_HEADS * HEAD_DIM;
        constexpr int64_t N_REP = NUM_HEADS / NUM_KV_HEADS;

        Tensor int_scalar(Operations &ops, int64_t value) {
            return ops.insert<Functional::Const>(Weight{"", {}, static_cast<double>(value)}, false,
                                                 Mln::DType::Int32);
        }

        Tensor apply_rope(Operations &ops, const Tensor &x, const Tensor &cos, const Tensor &sin,
                          int64_t heads, int64_t sequence, int64_t dim) {
            const int64_t rotary = ROTARY_DIM;
            Tensor rotated = ops.insert<Functional::Slice>(
                x, std::vector<int64_t>{0, 0, 0}, std::vector<int64_t>{heads, sequence, rotary},
                std::vector<int64_t>{1, 1, 1});
            Tensor broadcast_cos = ops.insert<Functional::Broadcast>(
                cos, std::vector<int64_t>{heads, sequence, rotary}, std::vector<int64_t>{1, 2});
            Tensor broadcast_sin = ops.insert<Functional::Broadcast>(
                sin, std::vector<int64_t>{heads, sequence, rotary}, std::vector<int64_t>{1, 2});

            const int64_t half = rotary / 2;
            const std::vector<int64_t> ones{1, 1, 1};
            const std::vector<int64_t> first_begin{0, 0, 0};
            const std::vector<int64_t> first_end{heads, sequence, half};
            const std::vector<int64_t> second_begin{0, 0, half};
            const std::vector<int64_t> second_end{heads, sequence, rotary};

            Tensor x1 = ops.insert<Functional::Slice>(rotated, first_begin, first_end, ones);
            Tensor x2 = ops.insert<Functional::Slice>(rotated, second_begin, second_end, ones);
            Tensor cos1 =
                ops.insert<Functional::Slice>(broadcast_cos, first_begin, first_end, ones);
            Tensor cos2 =
                ops.insert<Functional::Slice>(broadcast_cos, second_begin, second_end, ones);
            Tensor sin1 =
                ops.insert<Functional::Slice>(broadcast_sin, first_begin, first_end, ones);
            Tensor sin2 =
                ops.insert<Functional::Slice>(broadcast_sin, second_begin, second_end, ones);

            Tensor out1 = ops.insert<Functional::Subtract>(
                ops.insert<Functional::Multiply>(x1, std::move(cos1)),
                ops.insert<Functional::Multiply>(x2, std::move(sin1)));
            Tensor out2 = ops.insert<Functional::Add>(
                ops.insert<Functional::Multiply>(std::move(x2), std::move(cos2)),
                ops.insert<Functional::Multiply>(std::move(x1), std::move(sin2)));
            Tensor result =
                ops.insert<Functional::Concatenate>(std::move(out1), std::move(out2), 2);
            if (rotary == dim)
                return result;
            Tensor pass = ops.insert<Functional::Slice>(x, std::vector<int64_t>{0, 0, rotary},
                                                        std::vector<int64_t>{heads, sequence, dim},
                                                        std::vector<int64_t>{1, 1, 1});
            return ops.insert<Functional::Concatenate>(std::move(result), std::move(pass), 2);
        }

        Tensor repeat_kv(Operations &ops, const Tensor &x, int64_t seq_len) {
            Tensor r = ops.insert<Functional::Broadcast>(
                x, std::vector<int64_t>{NUM_KV_HEADS, N_REP, seq_len, HEAD_DIM},
                std::vector<int64_t>{0, 2, 3}, x);
            return ops.insert<Functional::Reshape>(
                std::move(r), std::vector<int64_t>{NUM_HEADS, seq_len, HEAD_DIM});
        }
    }

    Attention::Attention() {
        name = "self_attn";
        modules.insert<Linear>("q_proj", HIDDEN_SIZE, 2 * Q_DIM);
        modules.insert<Linear>("k_proj", HIDDEN_SIZE, KV_DIM);
        modules.insert<Linear>("v_proj", HIDDEN_SIZE, KV_DIM);
        modules.insert<Linear>("o_proj", Q_DIM, HIDDEN_SIZE);
        modules.insert<RMSNorm>("q_norm", HEAD_DIM, RMS_NORM_EPS, true);
        modules.insert<RMSNorm>("k_norm", HEAD_DIM, RMS_NORM_EPS, true);
    }

    Attention::Output Attention::forward(const Tensor &hidden, const Tensor &cos, const Tensor &sin,
                                         const Tensor &k_cache_in, const Tensor &v_cache_in,
                                         const Tensor &pos, const Tensor &mask) {
        const int64_t sequence = hidden.shape.size() == 1 ? 1 : hidden.shape[0];
        const int64_t context = mask.shape.back();
        Tensor projected_q = modules.forward<Linear>("q_proj", hidden);
        projected_q = operations.insert<Functional::Reshape>(
            std::move(projected_q), std::vector<int64_t>{sequence, NUM_HEADS, 2 * HEAD_DIM});
        Tensor q = operations.insert<Functional::Slice>(
            projected_q, std::vector<int64_t>{0, 0, 0},
            std::vector<int64_t>{sequence, NUM_HEADS, HEAD_DIM}, std::vector<int64_t>{1, 1, 1});
        Tensor gate = operations.insert<Functional::Slice>(
            std::move(projected_q), std::vector<int64_t>{0, 0, HEAD_DIM},
            std::vector<int64_t>{sequence, NUM_HEADS, 2 * HEAD_DIM}, std::vector<int64_t>{1, 1, 1});
        gate = operations.insert<Functional::Reshape>(std::move(gate),
                                                      std::vector<int64_t>{sequence, Q_DIM});
        q = operations.insert<Functional::Transpose>(std::move(q), std::vector<int64_t>{1, 0, 2});
        q = modules.forward<RMSNorm>("q_norm", q);
        q = apply_rope(operations, q, cos, sin, NUM_HEADS, sequence, HEAD_DIM);

        Tensor k = modules.forward<Linear>("k_proj", hidden);
        k = operations.insert<Functional::Reshape>(
            std::move(k), std::vector<int64_t>{sequence, NUM_KV_HEADS, HEAD_DIM});
        k = operations.insert<Functional::Transpose>(std::move(k), std::vector<int64_t>{1, 0, 2});
        k = modules.forward<RMSNorm>("k_norm", k);
        k = apply_rope(operations, k, cos, sin, NUM_KV_HEADS, sequence, HEAD_DIM);

        Tensor v = modules.forward<Linear>("v_proj", hidden);
        v = operations.insert<Functional::Reshape>(
            std::move(v), std::vector<int64_t>{sequence, NUM_KV_HEADS, HEAD_DIM});
        v = operations.insert<Functional::Transpose>(std::move(v), std::vector<int64_t>{1, 0, 2});

        Tensor pos_i = operations.insert<Functional::Convert>(pos, Mln::DType::Int32);
        Tensor zero = int_scalar(operations, 0);
        Tensor k_cache = operations.insert<Functional::DynamicUpdateSlice>(
            k_cache_in, std::move(k), std::vector<Tensor>{zero, pos_i, zero});
        Tensor v_cache = operations.insert<Functional::DynamicUpdateSlice>(
            v_cache_in, std::move(v), std::vector<Tensor>{zero, pos_i, zero});

        const std::vector<int64_t> begin{0, 0, 0};
        const std::vector<int64_t> end{NUM_KV_HEADS, context, HEAD_DIM};
        const std::vector<int64_t> stride{1, 1, 1};
        Tensor k_active = operations.insert<Functional::Slice>(k_cache, begin, end, stride, mask,
                                                               std::vector<int64_t>{1});
        Tensor v_active = operations.insert<Functional::Slice>(v_cache, begin, end, stride, mask,
                                                               std::vector<int64_t>{1});
        Tensor k_rep = repeat_kv(operations, k_active, context);
        Tensor v_rep = repeat_kv(operations, v_active, context);
        Tensor attn_out = operations.insert<Functional::Attention>(
            std::move(q), std::move(k_rep), std::move(v_rep),
            1.0 / std::sqrt(static_cast<double>(HEAD_DIM)), mask);
        attn_out = operations.insert<Functional::Transpose>(std::move(attn_out),
                                                            std::vector<int64_t>{1, 0, 2});
        attn_out = operations.insert<Functional::Reshape>(std::move(attn_out),
                                                          std::vector<int64_t>{sequence, Q_DIM});
        gate = operations.insert<Functional::Logistic>(std::move(gate));
        attn_out = operations.insert<Functional::Multiply>(std::move(attn_out), std::move(gate));
        Tensor out = modules.forward<Linear>("o_proj", attn_out);

        return Attention::Output{std::move(out), std::move(k_cache), std::move(v_cache)};
    }
}
