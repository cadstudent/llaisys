#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {

    // check device is same
    CHECK_SAME_DEVICE(out, in, weight);
    // check out and in are 2D tensor
    ASSERT(out->ndim() == 2, "Argmax: input out must be 2D tensor");
    ASSERT(in->ndim() == 2, "Argmax: input in must be 2D tensor");
    // check weight is 1D tensor
    ASSERT(weight->ndim() == 1, "Argmax: input weight must be 1D tensor");

    // Set current device and call implementation
    // llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());
    int batch_size = in->shape()[0];
    int feature_size = in->shape()[1];
    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::rms_norm(out->data(), in->data(), weight->data(), weight->dtype(), batch_size, feature_size, eps);
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
