//
//  MpmFluid.metal
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

#include "Mpm2DShared.h"
#include "UniformGrid2D.h"
#include "Utils.h"

using namespace metal;

struct MpmFluidParams {
    int particles_count;
    float timestep;
    float mass;
    float volume;
    
    float rest_density;
    float dynamic_viscosity;
    
    float eos_stiffness;
    float eos_power;
};

kernel void mpm_fluid_p2g1(device const float2* positions [[buffer(0)]],
                           device const float2* velocities [[buffer(1)]],
                           device const float2x2* Cs [[buffer(2)]],
                           device float* grid_ms [[buffer(3)]],
                           device float2* grid_vs [[buffer(4)]],
                           constant const UniformGrid2DParams& ug_params [[buffer(5)]],
                           constant const MpmFluidParams& mpm_params [[buffer(6)]],
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
    
    const auto C = Cs[p_i];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const int2 cell_i = p_cell + int2(i - 1, j - 1);
            if (!is_in_grid(ug_params, cell_i)) {
                continue;
            }
            const int cell_idx = to_cell_index(ug_params, cell_i);
            const float2 dpos = (castf(cell_i) + 0.5) * cell_size - pos_i;
            const float w = ws[i][0] * ws[j][1];
            
            const float mass_contrib = w * mpm_params.mass;
            fatomic_fetch_add(grid_ms + cell_idx, mass_contrib);
            f2atomic_fetch_add(grid_vs + cell_idx, mass_contrib * (velocities[p_i] + C * dpos));
        }
    }
}

kernel void mpm_fluid_p2g2(device const float2* positions [[buffer(0)]],
                           device const float2* velocities [[buffer(1)]],
                           device const float2x2* Cs [[buffer(2)]],
                           device float* grid_ms [[buffer(3)]],
                           device float2* grid_vs [[buffer(4)]],
                           constant const UniformGrid2DParams& ug_params [[buffer(5)]],
                           constant const MpmFluidParams& mpm_params [[buffer(6)]],
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
    
    float density = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const int2 cell_i = p_cell + int2(i - 1, j - 1);
            if (!is_in_grid(ug_params, cell_i)) {
                continue;
            }
            const int cell_idx = to_cell_index(ug_params, cell_i);
            
            const float w = ws[i][0] * ws[j][1];
            density += w * grid_ms[cell_idx];
        }
    }
    
    if (density <= 0) {
        return;
    }
    
    const float volume = mpm_params.mass / density;
    const float pressure = max(-0.1f, mpm_params.eos_stiffness * (pow(density / mpm_params.rest_density, mpm_params.eos_power) - 1.0f));
    
    float2x2 strain = Cs[p_i];
    {
        float strain_tr = trace(strain);
        strain[0][1] = strain_tr;
        strain[1][0] = strain_tr;
    }
    
    float2x2 stress(-pressure);
    stress += mpm_params.dynamic_viscosity * strain;
    const auto eq_16_term_0 = -volume * stress * mpm_params.timestep * 4.0f * inv_cell * inv_cell;
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const int2 cell_i = p_cell + int2(i - 1, j - 1);
            if (!is_in_grid(ug_params, cell_i)) {
                continue;
            }
            const int cell_idx = to_cell_index(ug_params, cell_i);
            const float2 dpos = (castf(cell_i) + 0.5) * cell_size - pos_i;
            const float w = ws[i][0] * ws[j][1];
            
            const float2 mv = w * eq_16_term_0 * dpos;
            f2atomic_fetch_add(grid_vs + cell_idx, mv);
        }
    }
}

kernel void mpm_fluid_advect(device float* grid_ms [[buffer(0)]],
                             device float2* grid_vs [[buffer(1)]],
                             constant const UniformGrid2DParams& ug_params [[buffer(2)]],
                             constant const MpmFluidParams& mpm_params [[buffer(3)]],
                             const uint tid [[thread_position_in_grid]]) {
    AdvectionParams adv_params;
    adv_params.timestep = mpm_params.timestep;
    run_advection(grid_ms, grid_vs, ug_params, adv_params, tid);
}

kernel void mpm_fluid_g2p(device float2* positions [[buffer(0)]],
                          device float2* velocities [[buffer(1)]],
                          device float2x2* Cs [[buffer(2)]],
                          device const float* grid_ms [[buffer(3)]],
                          device const float2* grid_vs [[buffer(4)]],
                          constant const UniformGrid2DParams& ug_params [[buffer(5)]],
                          constant const MpmFluidParams& mpm_params [[buffer(6)]],
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
}
