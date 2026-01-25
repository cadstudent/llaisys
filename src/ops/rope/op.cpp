#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // check
    CHECK_SAME_DEVICE(out, in);
    // check input
    ASSERT(in->ndim() == 3, "Rope: input must be 3D tensor");
    ASSERT(pos_ids->ndim() == 1, "Rope: pos_ids must be 1D tensor");
    ASSERT(out->shape() == in->shape(), "Rope: out must have same shape as in");
    // check dtype
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Rope: pos_ids must be int64 dtype");

    int64_t seq_len = in->shape()[0];
    int64_t head_size = in->shape()[1];
    int64_t d = in->shape()[2];
    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::rope(out->data(), in->data(), reinterpret_cast<int64_t *>(pos_ids->data()), in->dtype(), seq_len, head_size, d, theta);
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
