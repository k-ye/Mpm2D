//
//  Utils.h
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

#ifndef Utils_h
#define Utils_h

#include "UniformGrid2D.h"

using namespace metal;

float fatomic_fetch_add(device float *dest, const float operand);

void f2atomic_fetch_add(device float2* addr, float2 operand);


int2 casti(float2 f);

float2 castf(int2 i);

float2 sqr(float2 f);

int2 to_cell(constant const UniformGrid2DParams& c, thread const float2& pos);

int to_cell_index(constant const UniformGrid2DParams& c, thread const int2& cell);

int to_cell_index(constant const UniformGrid2DParams& c, thread const float2& pos);

int2 to_cell(constant const UniformGrid2DParams& c, int cell_index);

bool is_in_grid(constant const UniformGrid2DParams& c, const int2 cell);

float trace(thread float2x2& m);

float2x2 outer_product(float2 a, float2 b);

#endif /* Utils_h */
