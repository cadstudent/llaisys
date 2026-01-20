#include "embedding_cpu.hpp"
#include "../../../utils.hpp"

using namespace std;
// 核心 Embedding 查表函数
// 参数说明：
//   out_ptr:      输出指针 (2D, [batch_size, embed_dim])
//   weight_ptr:   权重指针 (2D, [vocab_size, embed_dim])
//   indices_ptr:  索引指针 (1D, [batch_size])
//   batch_size:   索引数量（output 的第一维大小）
//   embed_dim:    嵌入维度（weight/output 的第二维大小）
//   vocab_size:   词表大小（weight 的第一维大小，用于边界检查）
template <typename T>
void embedding_(T *out_ptr,
                T *weight_ptr,
                int64_t *indices_ptr,
                int64_t batch_size,
                int64_t embed_dim,
                int64_t vocab_size) {
    // 1. 边界检查（防止索引越界/空指针）
    if (out_ptr == nullptr || weight_ptr == nullptr || indices_ptr == nullptr) {
        throw std::runtime_error("embedding_: 输入/输出指针不能为空");
    }
    if (batch_size <= 0 || embed_dim <= 0 || vocab_size <= 0) {
        throw std::runtime_error("embedding_: 维度大小必须为正整数");
    }

    // 2. 遍历每个索引，复制对应行
    for (int64_t i = 0; i < batch_size; ++i) {
        // 获取当前索引值，并检查是否越界
        int64_t idx = indices_ptr[i];
        if (idx < 0 || idx >= vocab_size) {
            throw std::out_of_range("embedding_: 索引 " + std::to_string(idx) + " 超出词表范围 [0, " + std::to_string(vocab_size - 1) + "]");
        }

        // 计算 weight 中对应行的起始位置：行号 * 列维度
        T *weight_row_start = weight_ptr + idx * embed_dim;
        // 计算 output 中对应行的起始位置：行号 * 列维度
        T *out_row_start = out_ptr + i * embed_dim;

        // 3. 复制整行数据（从 weight 行到 output 行）
        for (int64_t j = 0; j < embed_dim; ++j) {
            out_row_start[j] = weight_row_start[j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(byte *out, byte *weight, int64_t *indices, llaisysDataType_t type, int64_t batch_size, int64_t embed_dim, int64_t vocab_size) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(weight), indices, batch_size, embed_dim, vocab_size);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(weight), indices, batch_size, embed_dim, vocab_size);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(weight), indices, batch_size, embed_dim, vocab_size);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu