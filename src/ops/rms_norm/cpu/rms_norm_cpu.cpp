#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

using namespace std;

// RMSNorm
//
// out: [batch_size, feature_size]
// input: [batch_size, feature_size]
// weight: [feature_size]
// batch_size: batch size
// feature_size: feature size

template <typename T>
void rms_norm_(T *out, T *input, T *weight, int batch_size, int feature_size, float eps) {
    for (int b = 0; b < batch_size; ++b) {
        float sum = 0;
        for (int f = 0; f < feature_size; ++f) {
            float in_f;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                in_f = llaisys::utils::cast<float>(input[b * feature_size + f]);
            } else {
                in_f = input[b * feature_size + f];
            }

            sum += in_f * in_f;
        }

        float rms = sqrt(sum / feature_size + eps);
        for (int f = 0; f < feature_size; ++f) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[b * feature_size + f] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(input[b * feature_size + f]) * llaisys::utils::cast<float>( weight[f]) / rms);
            } else {
                out[b * feature_size + f] = input[b * feature_size + f] * weight[f] / rms;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, std::byte *input, std::byte *weight, llaisysDataType_t type, int batch_size, int feature_size, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(input), reinterpret_cast<float *>(weight), batch_size, feature_size, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(input), reinterpret_cast<llaisys::bf16_t *>(weight), batch_size, feature_size, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(input), reinterpret_cast<llaisys::fp16_t *>(weight), batch_size, feature_size, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
