//
//  Mpm2DShared.h
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/26.
//  Copyright © 2020 zkk. All rights reserved.
//

#ifndef Mpm2DShared_h
#define Mpm2DShared_h

#include "UniformGrid2D.h"

struct AdvectionParams {
    float timestep;
    float2 gravity;
};

void run_advection(device float* grid_ms,
                   device float2* grid_vs,
                   constant const UniformGrid2DParams& ug_params,
                   thread const AdvectionParams& adv_params,
                   const uint tid);

struct G2pParams {
    int particles_count;
    float timestep;
};

struct G2pReslt {
    int particle_id = -1;
    float new_J_factor = 0;
};

G2pReslt run_g2p(device float2* positions,
                 device float2* velocities,
                 device metal::float2x2* Cs,
                 device const float* grid_ms,
                 device const float2* grid_vs,
                 constant const UniformGrid2DParams& ug_params,
                 thread const G2pParams& g2p_params,
                 const uint tid);

#endif /* Mpm2DShared_h */
