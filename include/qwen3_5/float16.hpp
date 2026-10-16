#pragma once

#include <bit>
#include <cstdint>

namespace Qwen35 {
    inline std::uint16_t to_float16(float value) noexcept {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        const std::uint32_t sign = (bits >> 16) & 0x8000;
        const std::uint32_t exponent = (bits >> 23) & 0xff;
        const std::uint32_t mantissa = bits & 0x7fffff;
        if (exponent == 0xff)
            return static_cast<std::uint16_t>(sign | 0x7c00 | (mantissa ? 0x0200 : 0));

        std::int32_t target_exponent = static_cast<std::int32_t>(exponent) - 127 + 15;
        if (target_exponent >= 31)
            return static_cast<std::uint16_t>(sign | 0x7c00);
        if (target_exponent <= 0) {
            if (target_exponent < -10)
                return static_cast<std::uint16_t>(sign);
            std::uint32_t rounded = (mantissa | 0x800000) >> (1 - target_exponent);
            rounded += 0x0fff + ((rounded >> 13) & 1);
            return static_cast<std::uint16_t>(sign | (rounded >> 13));
        }

        std::uint32_t rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
        if ((rounded & 0x800000) != 0) {
            rounded = 0;
            ++target_exponent;
        }
        return target_exponent >= 31
                   ? static_cast<std::uint16_t>(sign | 0x7c00)
                   : static_cast<std::uint16_t>(
                         sign | (static_cast<std::uint32_t>(target_exponent) << 10) |
                         (rounded >> 13));
    }
}
