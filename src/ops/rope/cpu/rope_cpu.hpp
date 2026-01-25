#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {
void rope(std::byte *out, std::byte *input, int64_t *pos_ids, llaisysDataType_t type, int seq_len, int head_size, int d, float rope_theta);
} // namespace llaisys::ops::cpu