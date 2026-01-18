#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    if (_meta.shape.empty()) {
        return true;
    }
    ptrdiff_t expected_stride = 1;
    // 从最后一个维度向前检查
    for (int i = _meta.shape.size() - 1; i >= 0; --i) {
        if (_meta.strides[i] != expected_stride) {
            return false;
        }
        // 计算前一个维度的期望步长
        expected_stride *= _meta.shape[i];
    }

    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != _meta.shape.size()) {
        throw std::invalid_argument("Permute order size must be equal to tensor shape size");
    }

    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    TensorMeta new_meta;
    new_meta.shape = std::move(new_shape);
    new_meta.strides = std::move(new_strides);
    new_meta.dtype = _meta.dtype;

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {

    // 1. 处理空张量或形状未变化的情况
    size_t original_elements = numel();
    if (original_elements == 0 || shape == _meta.shape) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    // 2. 计算原张量和新形状的总元素数
    size_t new_elements = 1;
    for (size_t s : shape) {
        new_elements *= s;
    }

    // 3. 验证元素数量是否匹配
    if (original_elements != new_elements) {
        std::stringstream ss;
        ss << "view shape mismatch: original elements=" << original_elements
           << ", new elements=" << new_elements;
        throw std::invalid_argument(ss.str());
    }

    // 4. 如果原张量是连续的，可以直接创建连续的新视图
    if (isContiguous()) {
        std::vector<ptrdiff_t> new_strides(shape.size());
        if (!shape.empty()) {
            new_strides.back() = 1;
            for (int i = shape.size() - 2; i >= 0; --i) {
                new_strides[i] = new_strides[i + 1] * shape[i + 1];
            }
        }

        TensorMeta new_meta;
        new_meta.dtype = _meta.dtype;
        new_meta.shape = shape;
        new_meta.strides = new_strides;

        return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
    }

    // 5. 处理非连续张量的情况，检查步长兼容性
    std::vector<size_t> orig_shape = _meta.shape;
    std::vector<ptrdiff_t> orig_strides = _meta.strides;

    size_t orig_idx = 0;
    size_t new_idx = 0;
    std::vector<ptrdiff_t> new_strides(shape.size(), 0);

    // 尝试合并/拆分维度以匹配新形状
    while (orig_idx < orig_shape.size() && new_idx < shape.size()) {
        ptrdiff_t orig_dim_size = orig_shape[orig_idx];
        ptrdiff_t orig_dim_stride = orig_strides[orig_idx];

        ptrdiff_t new_dim_size = shape[new_idx];

        // 如果当前维度大小匹配，直接使用原始步长
        if (orig_dim_size == new_dim_size) {
            new_strides[new_idx] = orig_dim_stride;
            orig_idx++;
            new_idx++;
            continue;
        }

        // 尝试合并原始维度以匹配新维度大小
        if (orig_dim_size < new_dim_size) {
            ptrdiff_t combined_size = orig_dim_size;
            ptrdiff_t combined_stride = orig_dim_stride;

            // 检查是否可以合并下一个原始维度
            while (orig_idx + 1 < orig_shape.size() && combined_size < new_dim_size) {
                ptrdiff_t next_size = orig_shape[orig_idx + 1];
                ptrdiff_t next_stride = orig_strides[orig_idx + 1];

                // 检查合并后的维度是否连续（步长是否匹配）
                if (combined_size * next_size > new_dim_size || next_stride != orig_dim_stride / combined_size) {
                    throw std::invalid_argument("view shape incompatible with tensor strides");
                }

                combined_size *= next_size;
                orig_idx++;
            }

            if (combined_size != new_dim_size) {
                throw std::invalid_argument("view shape incompatible with tensor strides");
            }

            new_strides[new_idx] = combined_stride;
            new_idx++;
            continue;
        }

        // 尝试拆分原始维度以匹配新维度大小
        if (orig_dim_size > new_dim_size) {
            if (orig_dim_size % new_dim_size != 0) {
                throw std::invalid_argument("view shape incompatible with tensor strides");
            }

            size_t split_size = orig_dim_size / new_dim_size;
            new_strides[new_idx] = orig_dim_stride;

            // 检查拆分后的维度是否连续
            if (new_idx + 1 >= shape.size() || shape[new_idx + 1] != split_size) {
                throw std::invalid_argument("view shape incompatible with tensor strides");
            }

            new_strides[new_idx + 1] = orig_dim_stride / split_size;
            orig_idx++;
            new_idx += 2;
            continue;
        }
    }

    // 检查是否处理完所有维度
    if (orig_idx != orig_shape.size() || new_idx != shape.size()) {
        throw std::invalid_argument("view shape incompatible with tensor strides");
    }

    // 6. 创建新的 TensorMeta 和 Tensor
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape = shape;
    new_meta.strides = new_strides;

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 1. 验证维度是否在有效范围内
    if (dim >= _meta.shape.size()) {
        std::stringstream ss;
        ss << "slice dimension out of range: dim=" << dim
           << ", ndim=" << _meta.shape.size();
        throw std::out_of_range(ss.str());
    }

    // 2. 验证切片范围是否有效
    if (start >= end) {
        throw std::invalid_argument("slice start must be less than end");
    }

    if (end > _meta.shape[dim]) {
        std::stringstream ss;
        ss << "slice end out of range: end=" << end
           << ", dim_size=" << _meta.shape[dim];
        throw std::out_of_range(ss.str());
    }

    // 3. 创建新的形状（仅修改指定维度）
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;

    // 4. 计算新的偏移量（字节）
    // 起始位置在该维度上的索引 * 该维度的步长 * 元素大小
    size_t element_size = utils::dsize(_meta.dtype);
    size_t new_offset = _offset + (start * _meta.strides[dim]) * element_size;

    // 5. 创建新的 TensorMeta（形状改变，步长保持不变）
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape = new_shape;
    new_meta.strides = _meta.strides;

    // 6. 返回新的 Tensor，共享底层 storage
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    const auto src = reinterpret_cast<const std::byte *>(src_);
    core::context().runtime().api()->memcpy_sync(
        this->data(),
        src,
        this->numel() * this->elementSize(),
        LLAISYS_MEMCPY_H2D);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
