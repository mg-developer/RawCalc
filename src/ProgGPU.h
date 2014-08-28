#ifndef _PROG_GPU_H
#define _PROG_GPU_H

#include <utility>
#include "ManagedMem.h"

struct ProcessParams {
    std::size_t count;
    ManagedPtr<int> data;

};

#endif
