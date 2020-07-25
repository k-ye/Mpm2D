//
//  Utils.metal
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

float fatomic_fetch_add(device float *dest, const float operand) {
  // natively.
  bool ok = false;
  float old_val = 0.0f;
  while (!ok) {
    old_val = *dest;
    float new_val = (old_val + operand);
    ok = atomic_compare_exchange_weak_explicit(
        (device atomic_int *)dest, (thread int *)(&old_val),
        *((thread int *)(&new_val)), metal::memory_order_relaxed,
        metal::memory_order_relaxed);
  }
  return old_val;
}

void f2atomic_fetch_add(device float2* addr, float2 operand) {
    // Metal doesn't seem to take the address of vectorized types. Had to manually
    // cast it to float. See an OpenCL related post:
    // https://groups.google.com/forum/#!topic/boost-compute/xJS05dkQEJk
    device float* base = reinterpret_cast<device float*>(addr);
    fatomic_fetch_add(base, operand[0]);
    fatomic_fetch_add(base + 1, operand[1]);
}

int2 casti(float2 f) {
    int2 result;
    result[0] = int(f[0]);
    result[1] = int(f[1]);
    return result;
}

float2 castf(int2 i) {
    float2 result;
    result[0] = float(i[0]);
    result[1] = float(i[1]);
    return result;
}

float2 sqr(float2 f) {
    return f * f;
}

int2 to_cell(constant const UniformGrid2DParams& c, thread const float2& pos) {
    return casti(pos / c.cell_size);
}

int to_cell_index(constant const UniformGrid2DParams& c, thread const int2& cell) {
    int result = cell[1] * c.grid[0];
    result += cell[0];
    return result;
}

int to_cell_index(constant const UniformGrid2DParams& c, thread const float2& pos) {
    const int2 cell = to_cell(c, pos);
    return to_cell_index(c, cell);
}

int2 to_cell(constant const UniformGrid2DParams& c, int cell_index) {
    const int gx = c.grid[0];
    int2 result;
    result[1] = cell_index / gx;
    result[0] = cell_index % gx;
    return result;
}

bool is_in_grid(constant const UniformGrid2DParams& c, const int2 cell) {
    return (0 <= cell[0] && cell[0] < c.grid[0] &&
            0 <= cell[1] && cell[1] < c.grid[1]);
}
