//
//  UniformGrid2DParams.swift
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

import Metal

struct UniformGrid2DParams {
    let boundary: Float2
    let grid: Int2
    let cellSize: Float
    let cellsCount: HMInt
    
    init(boundary: Float2, cellSize: Float) {
        self.boundary = boundary
        self.cellSize = cellSize
        
        let cellInv = 1.0 / cellSize
        let w = HMInt((boundary[0] * cellInv).rounded(.up))
        let h = HMInt((boundary[1] * cellInv).rounded(.up))
        self.grid = Int2(w, h)
        self.cellsCount = w * h
    }
}

class UniformGrid2DParamsHMPack {
    var params: UniformGrid2DParams {
        get { return _params }
    }
    
    var cellsCount: Int {
        get { return Int(params.cellsCount) }
    }
    
    private var _params: UniformGrid2DParams
    let buffer: MTLBuffer
    
    init(_ params: UniformGrid2DParams, _ device: MTLDevice) {
        self._params = params
        self.buffer = device.makeBuffer(bytes: &_params, length: MemoryLayout<UniformGrid2DParams>.stride, options: [])!
    }
}
