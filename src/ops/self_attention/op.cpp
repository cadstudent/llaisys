#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"
namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {

    int seqlen = q->shape()[0];
    int nhead = q->shape()[1];
    int nkvhead = k->shape()[1];
    int d = q->shape()[2];
    int dv = v->shape()[2];
    int total_len = k->shape()[0];

    switch (q->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(), seqlen, nhead, nkvhead, d, dv, total_len, scale);
        break;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        break;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
