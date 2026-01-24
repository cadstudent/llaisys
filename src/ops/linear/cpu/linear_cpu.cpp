#include "linear_cpu.hpp"
#include "../../../utils.hpp"

using namespace std;

// in [batch_size, in_features]
// weight [out_features, in_features]
// bias [out_features]
// out [batch_size, out_features]
//  batch_size : 批量大小
//  in_features : 输入特征数
//  out_features : 输出特征数
template <typename T>
void linear_(T *out, T *in, T *weight, T *bias, int batch_size, int in_features, int out_features) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_features; j++) {

            float sum;
            if (!bias) {
                sum = static_cast<float>(0);
            } else {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum = llaisys::utils::cast<float>(bias[j]);
                } else {
                    sum = bias[j];
                }
            }
            // (bias != nullptr) ? static_cast<float>(bias[j]) : static_cast<float>(0);
            for (int k = 0; k < in_features; k++) {
                float in_val;
                float weight_val;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    in_val = llaisys::utils::cast<float>(in[i * in_features + k]);
                    weight_val = llaisys::utils::cast<float>(weight[j * in_features + k]);
                } else {
                    in_val = in[i * in_features + k];
                    weight_val = weight[j * in_features + k];
                }
                sum += in_val * weight_val;
            }
            out[i * out_features + j] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(byte *out, byte *in, byte *weight, byte *bias, llaisysDataType_t type, int batch_size, int in_features, int out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in), reinterpret_cast<float *>(weight), reinterpret_cast<float *>(bias), batch_size, in_features, out_features);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(in), reinterpret_cast<llaisys::bf16_t *>(weight), reinterpret_cast<llaisys::bf16_t *>(bias), batch_size, in_features, out_features);
        break;
    case LLAISYS_DTYPE_F16:
        linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(in), reinterpret_cast<llaisys::fp16_t *>(weight), reinterpret_cast<llaisys::fp16_t *>(bias), batch_size, in_features, out_features);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
