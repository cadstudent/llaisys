#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, std::byte *input, std::byte *weight, llaisysDataType_t type, int batch_size, int feature_size, float eps);
}