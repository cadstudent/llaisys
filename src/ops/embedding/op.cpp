#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // check device is same
    CHECK_SAME_DEVICE(out, index, weight);

    // check out and weight are 2D tensor
    ASSERT(out->ndim() == 2, "Argmax: input out must be 2D tensor");
    ASSERT(weight->ndim() == 2, "Argmax: input weight must be 2D tensor");
    // check index is 1D tensor
    ASSERT(index->ndim() == 1, "Argmax: input index must be 1D tensor");
    // check data type
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Argmax: idx must be int64 dtype");
    ASSERT(out->dtype() == weight->dtype(), "Argmax: out and weight must have same dtype");

    // maybe need isContiguous check?
    int64_t batch_size = static_cast<int64_t>(index->shape()[0]);
    int64_t embed_dim = static_cast<int64_t>(weight->shape()[1]);
    int64_t vocab_size = static_cast<int64_t>(weight->shape()[0]);

    // Set current device and call implementation
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::embedding(out->data(), weight->data(), reinterpret_cast<int64_t *>(index->data()), weight->dtype(), batch_size, embed_dim, vocab_size);
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
