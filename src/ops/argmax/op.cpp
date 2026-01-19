#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"

using namespace std;
namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // 检查输入是1D张量
    ASSERT(vals->ndim() == 1, "Argmax: input vals must be 1D tensor");

    // 检查输出张量形状（应为包含单个元素的1D张量）
    ASSERT(max_idx->shape() == std::vector<size_t>({1}), "Argmax: max_idx must be 1D tensor with 1 element");
    ASSERT(max_val->shape() == std::vector<size_t>({1}), "Argmax: max_val must be 1D tensor with 1 element");

    // 检查数据类型
    ASSERT(max_val->dtype() == vals->dtype(), "Argmax: max_val and vals must have same dtype");
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be int64 dtype");

    // 检查连续性
    ASSERT(vals->isContiguous(), "Argmax: vals tensor must be contiguous");
    ASSERT(max_idx->isContiguous(), "Argmax: max_idx tensor must be contiguous");
    ASSERT(max_val->isContiguous(), "Argmax: max_val tensor must be contiguous");

    // CPU设备处理
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }
    // 设置当前设备
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
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
