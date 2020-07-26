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

#endif /* Mpm2DShared_h */
