#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

using namespace std;

// rope
// out: [seq_len, head_size, d]
// input: [seq_len, head_size, d]
// pos_ids: [seq_len]

template <typename T>
void rope_(T *out, T *input, int64_t *pos_ids, int seq_len, int head_size, int d, float rope_theta) {
    // pre compute
    const int half_d = d / 2;
    float *theta_denominator = new float[half_d];
    for (int j = 0; j < half_d; ++j) {
        const float exponent = 2.0f * j / d;
        theta_denominator[j] = std::pow(rope_theta, exponent);
    }

    for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
        const int64_t pos = pos_ids[seq_idx];

        for (int head_idx = 0; head_idx < head_size; ++head_idx) {

            const int base_offset = seq_idx * head_size * d + head_idx * d;

            for (int j = 0; j < half_d; ++j) {
                // 计算当前二维组的两个维度索引
                // PyTorch实现是将特征维度分成前半部分和后半部分，而不是按奇偶分组
                const int dim1 = base_offset + j;          // 前半部分维度（a_i,j）
                const int dim2 = base_offset + j + half_d; // 后半部分维度（b_i,j）

                // ===================== 4. 计算旋转角度 phi =====================
                // phi = pos / theta^(2j/d)
                const float phi = static_cast<float>(pos) / theta_denominator[j];

                // ===================== 5. 计算cos和sin值 =====================
                const float cos_phi = std::cos(phi);
                const float sin_phi = std::sin(phi);

                // ===================== 6. 读取原始输入值 =====================
                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(input[dim1]); // a_i,j
                    b = llaisys::utils::cast<float>(input[dim2]); // b_i,j
                } else {
                    a = input[dim1]; // a_i,j
                    b = input[dim2]; // b_i,j
                }

                // ===================== 7. 应用RoPE旋转公式 =====================
                // a' = a*cos(phi) - b*sin(phi)
                out[dim1] = llaisys::utils::cast<T>(a * cos_phi - b * sin_phi);
                // b' = b*cos(phi) + a*sin(phi)
                out[dim2] = llaisys::utils::cast<T>(b * cos_phi + a * sin_phi);
            }
        }
    }

    // ===================== 8. 释放临时内存 =====================
    delete[] theta_denominator;
}

namespace llaisys::ops::cpu {
void rope(byte *out, byte *input, int64_t *pos_ids, llaisysDataType_t type, int seq_len, int head_size, int d, float rope_theta) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(input), pos_ids, seq_len, head_size, d, rope_theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(input), pos_ids, seq_len, head_size, d, rope_theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(input), pos_ids, seq_len, head_size, d, rope_theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu