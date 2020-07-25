//
//  Mpm2DShared.metal
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/26.
//  Copyright © 2020 zkk. All rights reserved.
//

#include <metal_stdlib>
#include "Mpm2DShared.h"
#include "Utils.h"

using namespace metal;

void run_advection(device float* grid_ms,
                   device float2* grid_vs,
                   constant const UniformGrid2DParams& ug_params,
                   thread const AdvectionParams& adv_params,
                   const uint tid) {
    const int c_idx = (int)tid;
    if (c_idx >= ug_params.cells_count) {
        return;
    }
    
    const float c_mass = grid_ms[c_idx];
    if (c_mass <= 0.0) {
        return;
    }
    
    float2 c_vel = grid_vs[c_idx];
    c_vel /= c_mass;
    c_vel += adv_params.timestep * float2(0, -9.8f);  // gravity
    
    constexpr int kBumper = 2;
    const auto c_i = to_cell(ug_params, c_idx);
    if (c_i[0] < kBumper && c_vel[0] < 0) {
        c_vel[0] = 0;
    }
    if (c_i[0] >= (ug_params.grid[0] - kBumper) && c_vel[0] > 0) {
        c_vel[0] = 0;
    }
    if (c_i[1] < kBumper && c_vel[1] < 0) {
        c_vel[1] = 0;
    }
    if (c_i[1] >= (ug_params.grid[1] - kBumper) && c_vel[1] > 0) {
        c_vel[1] = 0;
    }
    
    grid_vs[c_idx] = c_vel;
}
