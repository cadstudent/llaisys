#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *input, std::byte *weight, std::byte *bias, llaisysDataType_t type, int batch_size, int in_features, int out_features);
} // namespace llaisys::ops::cpu