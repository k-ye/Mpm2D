//
//  UniformGrid2D.h
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

#ifndef UniformGrid2D_h
#define UniformGrid2D_h

struct UniformGrid2DParams {
    float2 boundary;
    int2 grid;
    float cell_size;
    int cells_count;
};

#endif /* UniformGrid2D_h */
