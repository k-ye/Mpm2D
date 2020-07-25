//
//  Mpm2D.metal
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

#include <metal_stdlib>
#include "UniformGrid2D.h"
#include "Utils.h"

using namespace metal;

struct Mpm2DParams {
    int particles_count;
    float timestep;
    float mass;
    float volume;
    float E;
};

kernel void p2g(device const float2* positions [[buffer(0)]],
                device const float2* velocities [[buffer(1)]],
                device const float2x2* Cs [[buffer(2)]],
                device const float* Js [[buffer(3)]],
                device float* grid_ms [[buffer(4)]],
                device float2* grid_vs [[buffer(5)]],
                constant const UniformGrid2DParams& ug_params [[buffer(6)]],
                constant const Mpm2DParams& mpm_params [[buffer(7)]],
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

kernel void advect(device float* grid_ms [[buffer(0)]],
                   device float2* grid_vs [[buffer(1)]],
                   constant const UniformGrid2DParams& ug_params [[buffer(2)]],
                   constant const Mpm2DParams& mpm_params [[buffer(3)]],
                   const uint tid [[thread_position_in_grid]]) {
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
    c_vel += mpm_params.timestep * float2(0, -20.0f);  // gravity
    
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

float2x2 outer_product(float2 a, float2 b) {
    // Column major...
    float2x2 m(0);
    m[0][0] = a[0] * b[0];
    m[1][0] = a[0] * b[1];
    m[0][1] = a[1] * b[0];
    m[1][1] = a[1] * b[1];
    return m;
}

float trace(thread float2x2& m) {
    return m[0][0] + m[1][1];
}

kernel void g2p(device float2* positions [[buffer(0)]],
                device float2* velocities [[buffer(1)]],
                device float2x2* Cs [[buffer(2)]],
                device float* Js [[buffer(3)]],
                device const float* grid_ms [[buffer(4)]],
                device const float2* grid_vs [[buffer(5)]],
                constant const UniformGrid2DParams& ug_params [[buffer(6)]],
                constant const Mpm2DParams& mpm_params [[buffer(7)]],
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
    
    const float dt = mpm_params.timestep;
    velocities[p_i] = new_v;
    positions[p_i] += new_v * dt;
    Cs[p_i] = new_C;
    Js[p_i] *= (1.0f + dt * trace(new_C));
}
