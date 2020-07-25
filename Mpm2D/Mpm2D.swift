//
//  Mpm2D.swift
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

import Metal
import simd

fileprivate let kGravityStrength = Float(9.81)
fileprivate let kClearGridKernel = "clear_grid"
fileprivate let kP2gKernel = "p2g"
fileprivate let kAdvectKernel = "advect"
fileprivate let kG2pKernel = "g2p"

struct Mtl1DThreadGridParams {
    let threadsPerGrid: Int
    let threadsPerGroup: Int
    
    init(threadsPerGrid: Int, threadsPerGroup: Int) {
        self.threadsPerGrid = threadsPerGrid
        self.threadsPerGroup = threadsPerGroup
    }
}

class Mpm2D: ParticlesProvider, ParticlesInitializer {
    fileprivate struct Params {
        var particlesCount: HMInt = .zero
        var timestep: Float = 1.0 / 1e4
        var mass: Float = .zero
        var volume: Float = .zero
        var E: Float = 400.0
    }
    
    class Builder {
        fileprivate var params = Params()
        fileprivate var ug: UniformGrid2DParamsHMPack!
        fileprivate var itersCount: Int = .zero
        
        var volume: Float {
            get { return powf(ug.params.cellSize, 2.0) * 0.25 }
        }
        
        func set(_ ugParamsPack: UniformGrid2DParamsHMPack) -> Builder {
            self.ug = ugParamsPack
            return self
        }
        
        func set(itersCount: Int) -> Builder {
            self.itersCount = itersCount
            return self
        }
        
        func set(particlesCount: Int) -> Builder {
            self.params.particlesCount = Int32(particlesCount)
            return self
        }
        
        func set(timestep: Float) -> Builder {
            self.params.timestep = timestep
            return self
        }
        
        func set(rho: Float) -> Builder {
            self.params.mass = rho * volume
            return self
        }
        
        func set(E: Float) -> Builder {
            self.params.E = E
            return self
        }
        
        func build(_ device: MTLDevice) -> Mpm2D {
            self.params.volume = volume
            return Mpm2D(self, device)
        }
    }
    
    let particlesCount: Int
    var positionsBuffer: MTLBuffer {
        get { return _positionsBuffer }
    }
    
    private var ugPack: UniformGrid2DParamsHMPack
    
    private let itersCount: Int
    private var mpmParams: Params
    private let mpmParamsBuffer: MTLBuffer
    
    
    private let _positionsBuffer: MTLBuffer
    private let velocitiesBuffer: MTLBuffer
    private let CsBuffer: MTLBuffer
    private let JsBuffer: MTLBuffer
    private let gridMsBuffer: MTLBuffer
    private let gridVsBuffer: MTLBuffer
    private var gravityBuffer: MTLBuffer!
    private var kernelPipelineStates = [String: MTLComputePipelineState]()
    
    fileprivate init(_ b: Builder, _ device: MTLDevice) {
        self.particlesCount = Int(b.params.particlesCount)
        self.itersCount = b.itersCount
        
        self.ugPack = b.ug
        
        self.mpmParams = b.params
        mpmParamsBuffer = device.makeBuffer(bytes: &mpmParams, length: MemoryLayout<Params>.stride, options: [])!
        
        _positionsBuffer = device.makeBuffer(length: particlesCount * MemoryLayout<Float2>.stride, options: [])!
        velocitiesBuffer = device.makeBuffer(length: particlesCount * MemoryLayout<Float2>.stride, options: [])!
        CsBuffer = device.makeBuffer(length: particlesCount * MemoryLayout<simd_float2x2>.stride, options: [])!
        let Js = [Float](repeating: 1.0, count: particlesCount)
        JsBuffer = device.makeBuffer(bytes: Js, length: particlesCount * MemoryLayout<Float>.stride, options: [])!
        
        let cellsCount = ugPack.cellsCount
        gridMsBuffer = device.makeBuffer(length: cellsCount * MemoryLayout<Float>.stride, options: [])!
        gridVsBuffer = device.makeBuffer(length: cellsCount * MemoryLayout<Float2>.stride, options: [])!
        
        var initGravity = Float2(0, -kGravityStrength)
        gravityBuffer = device.makeBuffer(bytes: &initGravity, length: MemoryLayout<Float2>.stride, options: [])!
        
        initKernelPipelineStates(device)
    }
    
    private func initKernelPipelineStates(_ device: MTLDevice) {
        let lib = device.makeDefaultLibrary()!
        for kernelName in [
            kClearGridKernel,
            kP2gKernel,
            kAdvectKernel,
            kG2pKernel,
            ] {
                let kernel = lib.makeFunction(name: kernelName)!
                kernelPipelineStates[kernelName] = try! device.makeComputePipelineState(function: kernel)
        }
    }
    
    func initParticle(i: Int, pos: Float2, vel: Float2) {
        _positionsBuffer.toMutablePtr(type: Float2.self, count: particlesCount)[i] = pos
        velocitiesBuffer.toMutablePtr(type: Float2.self, count: particlesCount)[i] = vel
    }
    
    func set(gravity: Float2) {
        let g = normalize(gravity) * kGravityStrength
        gravityBuffer.toMutablePtr(type: Float2.self, count: 1)[0] = g
    }
    
    func update(_ commandBuffer: MTLCommandBuffer) {
        for _ in 0..<itersCount {
            clearGrid(commandBuffer)
            p2g(commandBuffer)
            advect(commandBuffer)
            g2p(commandBuffer)
        }
    }
    
    private func clearGrid(_ commandBuffer: MTLCommandBuffer) {
//        let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
//        blitEncoder.fill(buffer: gridMsBuffer, range: 0..<gridMsBuffer.length, value: 0)
//        blitEncoder.fill(buffer: gridVsBuffer, range: 0..<gridVsBuffer.length, value: 0)
//        blitEncoder.endEncoding()
        
        
//        device float* grid_ms [[buffer(0)]],
//        device float2* grid_vs [[buffer(1)]],
//        constant const UniformGrid2DParams& ug_params [[buffer(2)]],
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        let pipelineState = kernelPipelineStates[kClearGridKernel]!
        commandEncoder.setComputePipelineState(pipelineState)
        let buffers = [
            gridMsBuffer,
            gridVsBuffer,
            ugPack.buffer,
        ]
        let tgParams = Mtl1DThreadGridParams(threadsPerGrid: ugPack.cellsCount, threadsPerGroup: pipelineState.maxTotalThreadsPerThreadgroup)
        finish(commandEncoder, buffers, tgParams)
    }
    
    private func p2g(_ commandBuffer: MTLCommandBuffer) {
//        device const float2* positions [[buffer(0)]],
//        device const float2* velocities [[buffer(1)]],
//        device const float2x2* Cs [[buffer(2)]],
//        device const float* Js [[buffer(3)]],
//        device float* grid_ms [[buffer(4)]],
//        device float2* grid_vs [[buffer(5)]],
//        constant const UniformGrid2DParams& ug_params [[buffer(6)]],
//        constant const Mpm2DParams& mpm_params [[buffer(7)]],
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        let pipelineState = kernelPipelineStates[kP2gKernel]!
        commandEncoder.setComputePipelineState(pipelineState)
        let buffers = [
            positionsBuffer,
            velocitiesBuffer,
            CsBuffer,
            JsBuffer,
            gridMsBuffer,
            gridVsBuffer,
            ugPack.buffer,
            mpmParamsBuffer,
        ]
        let tgParams = Mtl1DThreadGridParams(threadsPerGrid: particlesCount, threadsPerGroup: pipelineState.maxTotalThreadsPerThreadgroup)
        finish(commandEncoder, buffers, tgParams)
    }
    
    private func advect(_ commandBuffer: MTLCommandBuffer) {
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        let pipelineState = kernelPipelineStates[kAdvectKernel]!
        commandEncoder.setComputePipelineState(pipelineState)
//        device float* grid_ms [[buffer(0)]],
//        device float2* grid_vs [[buffer(1)]],
//        constant const UniformGrid2DParams& ug_params [[buffer(2)]],
//        constant const Mpm2DParams& mpm_params [[buffer(3)]],
        let buffers = [
            gridMsBuffer,
            gridVsBuffer,
            ugPack.buffer,
            mpmParamsBuffer,
        ]
        let tgParams = Mtl1DThreadGridParams(threadsPerGrid: ugPack.cellsCount, threadsPerGroup: pipelineState.maxTotalThreadsPerThreadgroup)
        finish(commandEncoder, buffers, tgParams)
    }
    
    private func g2p(_ commandBuffer: MTLCommandBuffer) {
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        let pipelineState = kernelPipelineStates[kG2pKernel]!
        commandEncoder.setComputePipelineState(pipelineState)
//        device float2* positions [[buffer(0)]],
//        device float2* velocities [[buffer(1)]],
//        device float2x2* Cs [[buffer(2)]],
//        device float* Js [[buffer(3)]],
//        device const float* grid_ms [[buffer(4)]],
//        device const float2* grid_vs [[buffer(5)]],
//        constant const UniformGrid2DParams& ug_params [[buffer(6)]],
//        constant const Mpm2DParams& mpm_params [[buffer(7)]],
        let buffers = [
            positionsBuffer,
            velocitiesBuffer,
            CsBuffer,
            JsBuffer,
            gridMsBuffer,
            gridVsBuffer,
            ugPack.buffer,
            mpmParamsBuffer,
        ]
        let tgParams = Mtl1DThreadGridParams(threadsPerGrid: particlesCount, threadsPerGroup: pipelineState.maxTotalThreadsPerThreadgroup)
        finish(commandEncoder, buffers, tgParams)
    }
    
    private func finish(_ commandEncoder: MTLComputeCommandEncoder, _ buffers: [MTLBuffer], _ tgp: Mtl1DThreadGridParams) {
        guard buffers.count > 0 else { fatalError("Empty buffers") }
        commandEncoder.setBuffers(buffers, offsets: [Int](repeating: 0, count: buffers.count), range: 0..<buffers.count)
        commandEncoder.dispatchThreads(MTLSizeMake(tgp.threadsPerGrid, 1, 1), threadsPerThreadgroup: MTLSizeMake(tgp.threadsPerGroup, 1, 1))
        commandEncoder.endEncoding()
    }
}
