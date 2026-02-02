#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace std;

float swish(const float &x) {

    if (x > 20.0f) {

        return x;
    } else if (x < -20.0f) {

        return 0.0f;
    } else {
        const float exp_val = std::exp(-x);
        return x / (1.0f + exp_val);
    }
}

// SwiGLU 核心实现（逐元素计算）
template <typename T>
void swiglu_(T *out, T *gate, T *up, int seqlen, int intermediate_size) {
    // ===================== 1. 边界检查 =====================
    if (out == nullptr || gate == nullptr || up == nullptr) {
        throw std::runtime_error("swiglu_: 输出/输入指针不能为空");
    }
    if (seqlen <= 0 || intermediate_size <= 0) {
        throw std::runtime_error("swiglu_: 维度不合法（seqlen=" + std::to_string(seqlen) + ", intermediate_size=" + std::to_string(intermediate_size) + "）");
    }

    // ===================== 2. 逐元素计算 SwiGLU =====================
    const int total_elems = seqlen * intermediate_size;
    for (int idx = 0; idx < total_elems; ++idx) {
        // 步骤1：计算 gate 的 Swish 激活
        float swish_gate, up_val;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            swish_gate = swish(llaisys::utils::cast<float>(gate[idx]));
            up_val = llaisys::utils::cast<float>(up[idx]);
        } else {
            swish_gate = swish(gate[idx]);
            up_val = up[idx];
        }

        out[idx] = llaisys::utils::cast<T>(up_val * swish_gate);
    }
}

namespace llaisys::ops::cpu {
// 对外暴露的 SwiGLU 实现
void swiglu(byte *out, byte *gate, byte *up,
            llaisysDataType_t type, int seqlen, int intermediate_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32: {
        float *out_ptr = reinterpret_cast<float *>(out);
        float *gate_ptr = reinterpret_cast<float *>(gate);
        float *up_ptr = reinterpret_cast<float *>(up);
        swiglu_(out_ptr, gate_ptr, up_ptr, seqlen, intermediate_size);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        llaisys::bf16_t *out_ptr = reinterpret_cast<llaisys::bf16_t *>(out);
        llaisys::bf16_t *gate_ptr = reinterpret_cast<llaisys::bf16_t *>(gate);
        llaisys::bf16_t *up_ptr = reinterpret_cast<llaisys::bf16_t *>(up);
        swiglu_(out_ptr, gate_ptr, up_ptr, seqlen, intermediate_size);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        llaisys::fp16_t *out_ptr = reinterpret_cast<llaisys::fp16_t *>(out);
        llaisys::fp16_t *gate_ptr = reinterpret_cast<llaisys::fp16_t *>(gate);
        llaisys::fp16_t *up_ptr = reinterpret_cast<llaisys::fp16_t *>(up);
        swiglu_(out_ptr, gate_ptr, up_ptr, seqlen, intermediate_size);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu