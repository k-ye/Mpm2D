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
    c_vel += adv_params.timestep * adv_params.gravity;  // gravity
    
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

G2pReslt run_g2p(device float2* positions,
                 device float2* velocities,
                 device metal::float2x2* Cs,
                 device const float* grid_ms,
                 device const float2* grid_vs,
                 constant const UniformGrid2DParams& ug_params,
                 thread const G2pParams& g2p_params,
                 const uint tid) {
    G2pReslt result;
    
    const int p_i = (int)tid;
    if (p_i >= g2p_params.particles_count) {
        result.particle_id = -1;
        return result;
    }
    
    const float cell_size = ug_params.cell_size;
    const float inv_cell = 1.0f / cell_size;
    const float2 pos_i = positions[p_i];
    const int2 p_cell = to_cell(ug_params, pos_i);
    const float2 fx = pos_i * inv_cell - castf(p_cell) - 0.5f;
    const float2 ws[3] = {
        0.5f * sqr(0.5f - fx),
        0.75f - sqr(fx),
        0.5f * sqr(0.5f + fx),
    };
    
    float2 new_v(0.0f);
    float2x2 new_C(0.0f);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const int2 cell_i = p_cell + int2(i - 1, j - 1);
            if (!is_in_grid(ug_params, cell_i)) {
                continue;
            }
            const int cell_idx = to_cell_index(ug_params, cell_i);
            const float2 dpos = (castf(cell_i) + 0.5) * cell_size - pos_i;
            const float2 c_vel = grid_vs[cell_idx];
            const float w = ws[i][0] * ws[j][1];
            
            new_v += w * c_vel;
            new_C += w * outer_product(c_vel, dpos) * 4.0f * inv_cell * inv_cell;
        }
    }
    
    const float dt = g2p_params.timestep;
    velocities[p_i] = new_v;
    positions[p_i] += new_v * dt;
    Cs[p_i] = new_C;
    
    result.particle_id = p_i;
    result.new_J_factor = (1.0f + dt * trace(new_C));
    return result;

}
