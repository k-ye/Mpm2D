//
//  ParticlesRenderer2D.swift
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

import Metal
import UIKit

fileprivate let kVertKernel = "point_vertex"
fileprivate let kFragKernel = "point_fragment"
fileprivate let kClearColor = MTLClearColor(red: 0.2, green: 0.3, blue: 0.3, alpha: 1.0)

class ParticlesRenderer2D {
    class Builder {
        fileprivate var ug: UniformGrid2DParamsHMPack!
        
        func set(_ ug: UniformGrid2DParamsHMPack) -> Builder {
            self.ug = ug
            return self
        }
        
        func build(_ device: MTLDevice) -> ParticlesRenderer2D {
            return ParticlesRenderer2D(self, device)
        }
    }
    
    private var ugPack: UniformGrid2DParamsHMPack!
    private let renderPipelineState: MTLRenderPipelineState
    
    fileprivate init(_ b: Builder, _ device: MTLDevice) {
        self.ugPack = b.ug
        
        let defaultLib = device.makeDefaultLibrary()!
        let vertexFunc = defaultLib.makeFunction(name: kVertKernel)!
        let fragFunc = defaultLib.makeFunction(name: kFragKernel)!
        
        let pipelineStateDesc = MTLRenderPipelineDescriptor()
        pipelineStateDesc.vertexFunction = vertexFunc
        pipelineStateDesc.fragmentFunction = fragFunc
        
        let colorAttachment = pipelineStateDesc.colorAttachments[0]!
        colorAttachment.pixelFormat = .bgra8Unorm
        renderPipelineState = try! device.makeRenderPipelineState(descriptor: pipelineStateDesc)
        
    }
    
    func render(_ particles: ParticlesProvider, _ drawable: CAMetalDrawable, _ commandBuffer: MTLCommandBuffer) {
        let texture = drawable.texture
        let renderPassDesc = MTLRenderPassDescriptor()
        
        let colorAttachment = renderPassDesc.colorAttachments[0]!
        colorAttachment.texture = texture
        colorAttachment.loadAction = .clear
        colorAttachment.clearColor = kClearColor
        colorAttachment.storeAction = .store
        
        let commandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc)!
        commandEncoder.setRenderPipelineState(renderPipelineState)
        commandEncoder.setVertexBuffer(particles.positionsBuffer, offset: 0, index: 0)
        commandEncoder.setVertexBuffer(ugPack.buffer, offset: 0, index: 1)
        commandEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: particles.particlesCount)
        commandEncoder.endEncoding()
        
        commandBuffer.present(drawable)
    }
}
