#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

using namespace std;
// Causal Softmax
// template <typename T>
// void causal_softmax(T *A, int seqlen, int total_len, int nhead) {
void causal_softmax(float *A, int seqlen, int total_len, int nhead) {
    // A的形状：[nhead, seqlen, total_len]（按此顺序存储）
    const float INF = 1e9;
    for (int h = 0; h < nhead; ++h) {      // 遍历每个头
        for (int i = 0; i < seqlen; ++i) { // 遍历当前序列的每个token（行）
            const int row_offset = h * seqlen * total_len + i * total_len;
            // 1. 应用因果掩码：列j > i + (total_len - seqlen) 时设为 -INF
            // 这与PyTorch的tril(diagonal=S-L)行为一致，其中S=total_len, L=seqlen
            for (int j = 0; j < total_len; ++j) {
                if (j > i + (total_len - seqlen)) {
                    A[row_offset + j] = -INF;
                }
            }
            // 2. 计算行内最大值（数值稳定，避免exp溢出）
            float row_max = A[row_offset];
            for (int j = 0; j < total_len; ++j) {
                if (A[row_offset + j] > row_max) {
                    row_max = A[row_offset + j];
                }
            }
            // 3. 计算行内exp和
            float exp_sum = 0.0f;
            for (int j = 0; j < total_len; ++j) {
                float exp_val = exp(A[row_offset + j] - row_max);
                A[row_offset + j] = exp_val;
                exp_sum += exp_val;
            }
            // 4. 归一化（softmax核心）
            for (int j = 0; j < total_len; ++j) {
                A[row_offset + j] = A[row_offset + j] / exp_sum;
            }
        }
    }
}

template <typename T>
void self_attention_(T *attn_val, T *q, T *k, T *v, llaisysDataType_t type,
                     int seqlen, int nhead, int nkvhead, int d, int dv, int total_len, float scale) {
    // ===================== 1. 边界检查 =====================
    if (attn_val == nullptr || q == nullptr || k == nullptr || v == nullptr) {
        throw std::runtime_error("self_attention_: 指针不能为空");
    }
    if (seqlen <= 0 || nhead <= 0 || nkvhead <= 0 || d <= 0 || dv <= 0 || total_len < seqlen) {
        throw std::runtime_error("self_attention_: 维度不合法（seqlen=" + std::to_string(seqlen) + ", nhead=" + std::to_string(nhead) + ", nkvhead=" + std::to_string(nkvhead) + ", d=" + std::to_string(d) + ", dv=" + std::to_string(dv) + ", total_len=" + std::to_string(total_len) + "）");
    }

    // ===================== 2. 步骤1：计算 QK^T * scale（A矩阵） =====================
    // A矩阵形状：[nhead, seqlen, total_len]，先初始化全0
    float *A = new float[nhead * seqlen * total_len]();
    for (int h = 0; h < nhead; ++h) { // 遍历每个查询头
        // MQA/GQA适配：kvhead = h % nkvhead（多查询/分组查询注意力）
        const int kvh = h * nkvhead / nhead;
        for (int i = 0; i < seqlen; ++i) {        // 遍历当前序列的每个token（Q的行）
            for (int j = 0; j < total_len; ++j) { // 遍历总序列的每个token（K的行）
                // 计算 Q[i,h,:] · K[j,kvh,:]
                float dot_product = 0.0f;
                for (int idx = 0; idx < d; ++idx) {
                    // Q的索引：[i, h, idx] → i*nhead*d + h*d + idx
                    // K的索引：[j, kvh, idx] → j*nkvhead*d + kvh*d + idx
                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q[i * nhead * d + h * d + idx]);
                        k_val = llaisys::utils::cast<float>(k[j * nkvhead * d + kvh * d + idx]);
                    } else {
                        q_val = q[i * nhead * d + h * d + idx];
                        k_val = k[j * nkvhead * d + kvh * d + idx];
                    }

                    dot_product += q_val * k_val;
                }
                // 缩放并写入A矩阵：A[h, i, j] = dot_product * scale
                A[h * seqlen * total_len + i * total_len + j] = dot_product * scale;
            }
        }
    }

    // ===================== 3. 步骤2：Causal Softmax =====================
    causal_softmax(A, seqlen, total_len, nhead);

    // ===================== 4. 步骤3：计算 Softmax(A) · V =====================
    for (int h = 0; h < nhead; ++h) {                     // 遍历每个查询头
        const int kvh = h * nkvhead / nhead;              // MQA/GQA适配
        for (int i = 0; i < seqlen; ++i) {                // 遍历当前序列的每个token
            for (int dv_idx = 0; dv_idx < dv; ++dv_idx) { // 遍历V的维度
                // 输出索引：[i, h, dv_idx] → i*nhead*dv + h*dv + dv_idx
                int out_idx = i * nhead * dv + h * dv + dv_idx;
                float val = 0.0f;
                // 累加：sum_j (A[h,i,j] * V[j, kvh, dv_idx])
                for (int j = 0; j < total_len; ++j) {
                    // A的索引：[h, i, j]
                    float a_val = A[h * seqlen * total_len + i * total_len + j];
                    // V的索引：[j, kvh, dv_idx] → j*nkvhead*dv + kvh*dv + dv_idx
                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[j * nkvhead * dv + kvh * dv + dv_idx]);
                    } else {
                        v_val = v[j * nkvhead * dv + kvh * dv + dv_idx];
                    }
                    val += a_val * v_val;
                }
                // if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                //     attn_val[out_idx] = llaisys::utils::cast<T>(val);
                // } else {
                //     attn_val[out_idx] = static_cast<T>(val);
                attn_val[out_idx] = llaisys::utils::cast<T>(val);
            }
        }
    }

    // ===================== 5. 释放临时内存 =====================
    delete[] A;
}

namespace llaisys::ops::cpu {

// 对外暴露的自注意力实现
void self_attention(byte *attn_val, byte *q, byte *k, byte *v,
                    llaisysDataType_t type, int seqlen, int nhead, int nkvhead,
                    int d, int dv, int total_len, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        float *attn_val_ptr = reinterpret_cast<float *>(attn_val);
        float *q_ptr = reinterpret_cast<float *>(q);
        float *k_ptr = reinterpret_cast<float *>(k);
        float *v_ptr = reinterpret_cast<float *>(v);
        self_attention_(attn_val_ptr, q_ptr, k_ptr, v_ptr, type,
                        seqlen, nhead, nkvhead, d, dv, total_len, scale);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        llaisys::bf16_t *attn_val_ptr = reinterpret_cast<llaisys::bf16_t *>(attn_val);
        llaisys::bf16_t *q_ptr = reinterpret_cast<llaisys::bf16_t *>(q);
        llaisys::bf16_t *k_ptr = reinterpret_cast<llaisys::bf16_t *>(k);
        llaisys::bf16_t *v_ptr = reinterpret_cast<llaisys::bf16_t *>(v);
        self_attention_(attn_val_ptr, q_ptr, k_ptr, v_ptr, type,
                        seqlen, nhead, nkvhead, d, dv, total_len, scale);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        llaisys::fp16_t *attn_val_ptr = reinterpret_cast<llaisys::fp16_t *>(attn_val);
        llaisys::fp16_t *q_ptr = reinterpret_cast<llaisys::fp16_t *>(q);
        llaisys::fp16_t *k_ptr = reinterpret_cast<llaisys::fp16_t *>(k);
        llaisys::fp16_t *v_ptr = reinterpret_cast<llaisys::fp16_t *>(v);
        self_attention_(attn_val_ptr, q_ptr, k_ptr, v_ptr, type,
                        seqlen, nhead, nkvhead, d, dv, total_len, scale);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
