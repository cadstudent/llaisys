#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

using namespace std;
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    cout<<"linear"<<endl;
    //check device is same
    CHECK_SAME_DEVICE(out, in, weight);

    //check in and weight 2d tensor
    //....

    int batch_size = in->shape()[0];
    int in_features = in->shape()[1];
    int out_features = out->shape()[1];

    // Set current device and call implementation
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), batch_size, in_features, out_features);
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
