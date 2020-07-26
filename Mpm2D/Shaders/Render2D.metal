//
//  Render.metal
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

#include <metal_stdlib>
#include "UniformGrid2D.h"

using namespace metal;

struct VertexOut {
    float4 position [[ position ]];
    // diameter in pixels
    float diameter [[ point_size ]];
};

constant static constexpr float kSphereDiameter = 1.6f;

vertex VertexOut point_vertex(device const float2* positions [[buffer(0)]],
                              constant const UniformGrid2DParams& ug_params [[buffer(1)]],
                              uint vid [[vertex_id]]) {
    VertexOut result;
    const float2 world_pos = positions[vid];
    
    float4 ndc_pos(0.0f);
    ndc_pos[0] = -1.0f + (world_pos[0] / ug_params.boundary[0]) * 2.0f;
    ndc_pos[1] = -1.0f + (world_pos[1] / ug_params.boundary[1]) * 2.0f;
    ndc_pos[3] = 1.0f;
    
    result.position = ndc_pos;
    result.diameter = kSphereDiameter;

    return result;
}

fragment float4 point_fragment(VertexOut frag_data [[stage_in]],
                               const float2 point_coord [[point_coord]]) {
//    const float pc_x = point_coord.x * 2.0 - 1.0;
//    const float pc_y = 1.0 - point_coord.y * 2.0;
//    const float xy_len_sqr = pc_x * pc_x + pc_y * pc_y;
//    if (xy_len_sqr >= 1.0) {
//        discard_fragment();
//    }
    return float4(1.0f, 0.5f, 0.2f, 1.0f);
}
