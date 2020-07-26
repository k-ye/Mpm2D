//
//  Mpm88.metal
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

#include <metal_stdlib>

#include "Mpm2DShared.h"
#include "UniformGrid2D.h"
#include "Utils.h"

using namespace metal;

struct Mpm88Params {
    int particles_count;
    float timestep;
    float mass;
    float volume;
    float E;
};


kernel void mpm88_p2g(device const float2* positions [[buffer(0)]],
                      device const float2* velocities [[buffer(1)]],
                      device const float2x2* Cs [[buffer(2)]],
                      device const float* Js [[buffer(3)]],
                      device float* grid_ms [[buffer(4)]],
                      device float2* grid_vs [[buffer(5)]],
                      constant const UniformGrid2DParams& ug_params [[buffer(6)]],
                      constant const Mpm88Params& mpm_params [[buffer(7)]],
                      const uint tid [[thread_position_in_grid]]) {
    const int p_i = (int)tid;
    if (p_i >= mpm_params.particles_count) {
        return;
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
    
    const float stress = -mpm_params.timestep * mpm_params.volume
        * (Js[p_i] - 1.0f) * 4.0f * inv_cell * inv_cell * mpm_params.E;
    float2x2 affine(stress);
    affine += mpm_params.mass * Cs[p_i];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const int2 cell_i = p_cell + int2(i - 1, j - 1);
            if (!is_in_grid(ug_params, cell_i)) {
                continue;
            }
            const int cell_idx = to_cell_index(ug_params, cell_i);
            const float2 dpos = (castf(cell_i) + 0.5) * cell_size - pos_i;
            const float w = ws[i][0] * ws[j][1];
            fatomic_fetch_add(grid_ms + cell_idx, w * mpm_params.mass);
            
            const float2 mv = w * (mpm_params.mass * velocities[p_i] + affine * dpos);
            f2atomic_fetch_add(grid_vs + cell_idx, mv);
        }
    }
}

kernel void mpm88_advect(device float* grid_ms [[buffer(0)]],
                         device float2* grid_vs [[buffer(1)]],
                         constant const UniformGrid2DParams& ug_params [[buffer(2)]],
                         constant const Mpm88Params& mpm_params [[buffer(3)]],
                         constant const float2& gravity [[buffer(4)]],
                         const uint tid [[thread_position_in_grid]]) {
    AdvectionParams adv_params;
    adv_params.timestep = mpm_params.timestep;
    adv_params.gravity = gravity;
    run_advection(grid_ms, grid_vs, ug_params, adv_params, tid);
}

kernel void mpm88_g2p(device float2* positions [[buffer(0)]],
                      device float2* velocities [[buffer(1)]],
                      device float2x2* Cs [[buffer(2)]],
                      device float* Js [[buffer(3)]],
                      device const float* grid_ms [[buffer(4)]],
                      device const float2* grid_vs [[buffer(5)]],
                      constant const UniformGrid2DParams& ug_params [[buffer(6)]],
                      constant const Mpm88Params& mpm_params [[buffer(7)]],
                      const uint tid [[thread_position_in_grid]]) {
    G2pParams g2p_params;
    g2p_params.particles_count = mpm_params.particles_count;
    g2p_params.timestep = mpm_params.timestep;
    const auto result = run_g2p(positions, velocities, Cs, grid_ms, grid_vs, ug_params, g2p_params, tid);
    
    if (result.particle_id != -1) {
        Js[result.particle_id] *= result.new_J_factor;
    }
}
