#include "argmax_cpu.hpp"
#include "../../../utils.hpp"

template <typename T>
void argmax_(T *max_idx_ptr, T *max_val_ptr, T *vals, size_t numel) {

    size_t idx = 0;
    float val = 0.0f;
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        val = llaisys::utils::cast<float>(vals[0]);

    } else {
        val = vals[0];
    }

    for (size_t i = 1; i < numel; ++i) {
        float val_i;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            val_i = llaisys::utils::cast<float>(vals[i]);
        } else {
            val_i = vals[i];
        }
        if (val_i > val) {
            val = val_i;
            idx = i;
        }
    }

    *max_val_ptr = llaisys::utils::cast<T>(val);
    *max_idx_ptr =  llaisys::utils::cast<T>(idx);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx_ptr, std::byte *max_val_ptr, std::byte *vals_ptr, llaisysDataType_t type, size_t numel) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_(reinterpret_cast<float *>(max_idx_ptr), reinterpret_cast<float *>(max_val_ptr), reinterpret_cast<float *>(vals_ptr), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_(reinterpret_cast<llaisys::bf16_t *>(max_idx_ptr), reinterpret_cast<llaisys::bf16_t *>(max_val_ptr), reinterpret_cast<llaisys::bf16_t *>(vals_ptr), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_(reinterpret_cast<llaisys::fp16_t *>(max_idx_ptr), reinterpret_cast<llaisys::fp16_t *>(max_val_ptr), reinterpret_cast<llaisys::fp16_t *>(vals_ptr), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu