#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, std::byte *q, std::byte *k, std::byte *v,
                     llaisysDataType_t type, int seqlen, int nhead, int nkvhead,
                     int d, int dv, int total_len, float scale);
}