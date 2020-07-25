//
//  ParticlesProtocols.swift
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

import Metal

protocol ParticlesProvider: class {
    var particlesCount: Int { get }
    var positionsBuffer: MTLBuffer { get }
}

protocol ParticlesInitializer: class {
    func initParticle(i: Int, pos: Float2, vel: Float2)
}
