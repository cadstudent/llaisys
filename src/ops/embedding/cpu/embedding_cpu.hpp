#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *weight, int64_t *indices, llaisysDataType_t type, int64_t batch_size, int64_t embed_dim, int64_t vocab_size);
} // namespace llaisys::ops::cpu