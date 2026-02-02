#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    int seqlen = out->shape()[0];
    int intermediate_size = out->shape()[1];

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), seqlen, intermediate_size);
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
