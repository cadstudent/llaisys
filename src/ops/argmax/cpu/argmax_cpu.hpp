#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx_ptr, std::byte *max_val_ptr,  std::byte *vals, llaisysDataType_t type, size_t size);
}