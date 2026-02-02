#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu{
    void swiglu(std::byte *out, std::byte *gate, std::byte *up,
    llaisysDataType_t type, int seqlen, int intermediate_size);  
}
