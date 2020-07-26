//
//  Mpm2D.swift
//  Mpm2DSolver
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

import Metal
import simd

struct Mtl1DThreadGridParams {
    let threadsPerGrid: Int
    let threadsPerGroup: Int
    
    init(threadsPerGrid: Int, threadsPerGroup: Int) {
        self.threadsPerGrid = threadsPerGrid
        self.threadsPerGroup = threadsPerGroup
    }
}

protocol Mpm2DSolver: ParticlesInitializer, ParticlesProvider {
    func update(_ commandBuffer: MTLCommandBuffer)
    func set(gravity: Float2)
    func getTransferrable() -> Mpm2DTransferrableData
}

fileprivate let kGravityStrength = Float(9.81)

fileprivate func volumeOfCell(_ ug: UniformGrid2DParamsHMPack) -> Float {
    let x = ug.params.cellSize * 0.5
    return x * x
}

fileprivate typealias ComputePipelineStatesMap = [String:MTLComputePipelineState]

fileprivate func makeComputePipelineStatesMap(with kernelNames: [String], _ device: MTLDevice) -> ComputePipelineStatesMap {
    let lib = device.makeDefaultLibrary()!
    var m = ComputePipelineStatesMap()
    for name in kernelNames {
            let kernel = lib.makeFunction(name: name)!
            m[name] = try! device.makeComputePipelineState(function: kernel)
    }
    return m
}

fileprivate class Mpm2DSolverShared {
    let particlesCount: Int
    var ugPack: UniformGrid2DParamsHMPack!
    
    let positionsBuffer: MTLBuffer
    let velocitiesBuffer: MTLBuffer
    let CsBuffer: MTLBuffer
    let gridMsBuffer: MTLBuffer
    let gridVsBuffer: MTLBuffer
    var gravityBuffer: MTLBuffer!
    
    private var kernelPipelineStates: ComputePipelineStatesMap
    
    class Builder {
        var ug: UniformGrid2DParamsHMPack!
        var particlesCount: Int = .zero
        var kernelNames = [String]()
        var transferrable: Mpm2DTransferrableData?
        
        func build(_ device: MTLDevice) -> Mpm2DSolverShared {
            return Mpm2DSolverShared(self, device)
        }
    }
    
    private init(_ b: Builder, _ device: MTLDevice) {
        if let other = b.transferrable?.shared {
            self.particlesCount = other.particlesCount
            self.ugPack = other.ugPack
            
            positionsBuffer = other.positionsBuffer
            velocitiesBuffer = other.velocitiesBuffer
            CsBuffer = other.CsBuffer
            
            gridMsBuffer = other.gridMsBuffer
            gridVsBuffer = other.gridVsBuffer
            
            gravityBuffer = other.gravityBuffer
        } else {
            self.particlesCount = b.particlesCount
            self.ugPack = b.ug
            
            positionsBuffer = device.makeBuffer(length: particlesCount * MemoryLayout<Float2>.stride, options: [])!
            velocitiesBuffer = device.makeBuffer(length: particlesCount * MemoryLayout<Float2>.stride, options: [])!
            CsBuffer = device.makeBuffer(length: particlesCount * MemoryLayout<simd_float2x2>.stride, options: [])!
            
            let cellsCount = ugPack.cellsCount
            gridMsBuffer = device.makeBuffer(length: cellsCount * MemoryLayout<Float>.stride, options: [])!
            gridVsBuffer = device.makeBuffer(length: cellsCount * MemoryLayout<Float2>.stride, options: [])!
            
            var initGravity = Float2(0, -kGravityStrength)
            gravityBuffer = device.makeBuffer(bytes: &initGravity, length: MemoryLayout<Float2>.stride, options: [])!
        }
        
        kernelPipelineStates = makeComputePipelineStatesMap(with: b.kernelNames, device)
    }
    
    func initParticle(_ i: Int, _ pos: Float2, _ vel: Float2) {
        positionsBuffer.toMutablePtr(type: Float2.self, count: particlesCount)[i] = pos
        velocitiesBuffer.toMutablePtr(type: Float2.self, count: particlesCount)[i] = vel
    }
    
    func set(gravity: Float2) {
        let g = normalize(gravity) * kGravityStrength
        gravityBuffer.toMutablePtr(type: Float2.self, count: 1)[0] = g
    }
    
    func getKernel(_ name: String) -> MTLComputePipelineState {
        return kernelPipelineStates[name]!
    }
    
    func clearGrid(_ commandBuffer: MTLCommandBuffer) {
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
        blitEncoder.fill(buffer: gridMsBuffer, range: 0..<gridMsBuffer.length, value: 0)
        blitEncoder.fill(buffer: gridVsBuffer, range: 0..<gridVsBuffer.length, value: 0)
        blitEncoder.endEncoding()
    }
    
    func advect(name: String, _ paramsBuffer: MTLBuffer, _ commandBuffer: MTLCommandBuffer) {
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        let pipelineState = getKernel(name)
        commandEncoder.setComputePipelineState(pipelineState)
        let buffers = [
            gridMsBuffer,
            gridVsBuffer,
            ugPack.buffer,
            paramsBuffer,
            gravityBuffer!,
        ]
        let tgParams = Mtl1DThreadGridParams(threadsPerGrid: ugPack.cellsCount, threadsPerGroup: pipelineState.maxTotalThreadsPerThreadgroup)
        finish(commandEncoder, buffers, tgParams)
    }
    
    func finish(_ commandEncoder: MTLComputeCommandEncoder, _ buffers: [MTLBuffer], _ tgp: Mtl1DThreadGridParams) {
        guard buffers.count > 0 else { fatalError("Empty buffers") }
        commandEncoder.setBuffers(buffers, offsets: [Int](repeating: 0, count: buffers.count), range: 0..<buffers.count)
        commandEncoder.dispatchThreads(MTLSizeMake(tgp.threadsPerGrid, 1, 1), threadsPerThreadgroup: MTLSizeMake(tgp.threadsPerGroup, 1, 1))
        commandEncoder.endEncoding()
    }
}

class Mpm2DTransferrableData {
    fileprivate var shared: Mpm2DSolverShared
    
    fileprivate init(_ s: Mpm2DSolverShared) {
        self.shared = s
    }
}

// MPM88 Solver
// https://github.com/taichi-dev/taichi/blob/master/examples/mpm88.py
class Mpm88SolverBuilder {
    fileprivate var params = Mpm88Impl.Params()
    fileprivate var ug: UniformGrid2DParamsHMPack!
    fileprivate var itersCount: Int = .zero
    fileprivate var transferrable: Mpm2DTransferrableData?
    
    var volume: Float {
        get { return volumeOfCell(ug) }
    }
    
    func set(_ ugParamsPack: UniformGrid2DParamsHMPack) -> Mpm88SolverBuilder {
        self.ug = ugParamsPack
        return self
    }
    
    func set(itersCount: Int) -> Mpm88SolverBuilder {
        self.itersCount = itersCount
        return self
    }
    
    func set(particlesCount: Int) -> Mpm88SolverBuilder {
        self.params.particlesCount = Int32(particlesCount)
        return self
    }
    
    func set(timestep: Float) -> Mpm88SolverBuilder {
        self.params.timestep = timestep
        return self
    }
    
    func set(rho: Float) -> Mpm88SolverBuilder {
        self.params.mass = rho * volume
        return self
    }
    
    func set(E: Float) -> Mpm88SolverBuilder {
        self.params.E = E
        return self
    }
    
    func set(_ transferrable: Mpm2DTransferrableData?) -> Mpm88SolverBuilder {
        self.transferrable = transferrable
        return self
    }
    
    func build(_ device: MTLDevice) -> Mpm2DSolver {
        self.params.volume = volume
        return Mpm88Impl(self, device)
    }
}

fileprivate class Mpm88Impl: Mpm2DSolver {
    struct Params {
        var particlesCount: HMInt = .zero
        var timestep: Float = .zero
        var mass: Float = .zero
        var volume: Float = .zero
        var E: Float = .zero
    }
    
    static let kP2gKernel = "mpm88_p2g"
    static let kAdvectKernel = "mpm88_advect"
    static let kG2pKernel = "mpm88_g2p"
    
    var particlesCount: Int {
        get { return shared.particlesCount }
    }
    
    var positionsBuffer: MTLBuffer {
        get { return shared.positionsBuffer }
    }
    
    var params: Params
    let paramsBuffer: MTLBuffer
    
    let itersCount: Int
    let shared: Mpm2DSolverShared
    let JsBuffer: MTLBuffer
    
    init(_ b: Mpm88SolverBuilder, _ device: MTLDevice) {
        self.params = b.params
        self.paramsBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<Params>.stride, options: [])!
        self.itersCount = b.itersCount
        
        let sharedBuilder = Mpm2DSolverShared.Builder()
        if let tr = b.transferrable {
            sharedBuilder.transferrable = tr
        } else {
            sharedBuilder.particlesCount = Int(b.params.particlesCount)
            sharedBuilder.ug = b.ug
        }
        sharedBuilder.kernelNames = [
            Mpm88Impl.kP2gKernel,
            Mpm88Impl.kAdvectKernel,
            Mpm88Impl.kG2pKernel,
        ]
        self.shared = sharedBuilder.build(device)
        
        let Js = [Float](repeating: 1.0, count: shared.particlesCount)
        JsBuffer = device.makeBuffer(bytes: Js, length: shared.particlesCount * MemoryLayout<Float>.stride, options: [])!
    }
    
    func update(_ commandBuffer: MTLCommandBuffer) {
        for _ in 0..<itersCount {
            shared.clearGrid(commandBuffer)
            
            p2g(commandBuffer)
            advect(commandBuffer)
            g2p(commandBuffer)
        }
    }
    
    func initParticle(i: Int, pos: Float2, vel: Float2) {
        shared.initParticle(i, pos, vel)
    }
    
    func set(gravity: Float2) {
        shared.set(gravity: gravity)
    }
    
    func getTransferrable() -> Mpm2DTransferrableData {
        return Mpm2DTransferrableData(shared)
    }
    
    private func p2g(_ commandBuffer: MTLCommandBuffer) {
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        let pipelineState = shared.getKernel(Mpm88Impl.kP2gKernel)
        commandEncoder.setComputePipelineState(pipelineState)
        let buffers = [
            positionsBuffer,
            shared.velocitiesBuffer,
            shared.CsBuffer,
            JsBuffer,
            shared.gridMsBuffer,
            shared.gridVsBuffer,
            shared.ugPack.buffer,
            paramsBuffer,
        ]
        let tgParams = Mtl1DThreadGridParams(threadsPerGrid: particlesCount, threadsPerGroup: pipelineState.maxTotalThreadsPerThreadgroup)
        shared.finish(commandEncoder, buffers, tgParams)
    }
    
    private func advect(_ commandBuffer: MTLCommandBuffer) {
        shared.advect(name: Mpm88Impl.kAdvectKernel, paramsBuffer, commandBuffer)
    }
    
    private func g2p(_ commandBuffer: MTLCommandBuffer) {
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        let pipelineState = shared.getKernel(Mpm88Impl.kG2pKernel)
        commandEncoder.setComputePipelineState(pipelineState)
        let buffers = [
            positionsBuffer,
            shared.velocitiesBuffer,
            shared.CsBuffer,
            JsBuffer,
            shared.gridMsBuffer,
            shared.gridVsBuffer,
            shared.ugPack.buffer,
            paramsBuffer,
        ]
        let tgParams = Mtl1DThreadGridParams(threadsPerGrid: particlesCount, threadsPerGroup: pipelineState.maxTotalThreadsPerThreadgroup)
        shared.finish(commandEncoder, buffers, tgParams)
    }
}

// MPM Fluid
// https://nialltl.neocities.org/articles/mpm_guide.html
class MpmNialltlSolverBuilder {
    fileprivate var params = MpmNialltlImpl.Params()
    fileprivate var ug: UniformGrid2DParamsHMPack!
    fileprivate var itersCount: Int = .zero
    fileprivate var transferrable: Mpm2DTransferrableData?
    
    var volume: Float {
        get { return volumeOfCell(ug) }
    }
    
    func set(_ ugParamsPack: UniformGrid2DParamsHMPack) -> MpmNialltlSolverBuilder {
        self.ug = ugParamsPack
        return self
    }
    func set(itersCount: Int) -> MpmNialltlSolverBuilder {
        self.itersCount = itersCount
        return self
    }

    func set(particlesCount: Int) -> MpmNialltlSolverBuilder {
        self.params.particlesCount = Int32(particlesCount)
        return self
    }
    
    func set(timestep: Float) -> MpmNialltlSolverBuilder {
        self.params.timestep = timestep
        return self
    }
    
    func set(restDensity: Float) -> MpmNialltlSolverBuilder {
        self.params.restDensity = restDensity
        self.params.mass = restDensity * volume
        return self
    }
    
    func set(dynamicViscosity: Float) -> MpmNialltlSolverBuilder {
        self.params.dynamicViscosity = dynamicViscosity
        return self
    }
    
    func set(eosStiffness: Float) -> MpmNialltlSolverBuilder {
        self.params.eosStiffness = eosStiffness
        return self
    }
    
    func set(eosPower: Float) -> MpmNialltlSolverBuilder {
        self.params.eosPower = eosPower
        return self
    }
    
    func set(_ transferrable: Mpm2DTransferrableData?) -> MpmNialltlSolverBuilder {
        self.transferrable = transferrable
        return self
    }
    
    func build(_ device: MTLDevice) -> Mpm2DSolver {
        self.params.volume = volume
        return MpmNialltlImpl(self, device)
    }
}

fileprivate class MpmNialltlImpl: Mpm2DSolver {
    struct Params {
        var particlesCount: HMInt = .zero
        var timestep: Float = .zero
        var mass: Float = .zero
        var volume: Float = .zero
        // fluid parameters
        var restDensity: Float = .zero
        var dynamicViscosity: Float = .zero
        // equation of state
        var eosStiffness: Float = .zero
        var eosPower: Float = .zero
    }
    
    static let kP2g1Kernel = "nialltl_p2g1"
    static let kP2g2Kernel = "nialltl_p2g2"
    static let kAdvectKernel = "nialltl_advect"
    static let kG2pKernel = "nialltl_g2p"
    
    var particlesCount: Int {
        get { return shared.particlesCount }
    }
    
    var positionsBuffer: MTLBuffer {
        get { return shared.positionsBuffer }
    }
    
    var params: Params
    let paramsBuffer: MTLBuffer
    
    let itersCount: Int
    let shared: Mpm2DSolverShared

    init(_ b: MpmNialltlSolverBuilder, _ device: MTLDevice) {
        self.params = b.params
        self.paramsBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<Params>.stride, options: [])!
        self.itersCount = b.itersCount
        
        let sharedBuilder = Mpm2DSolverShared.Builder()
        if let tr = b.transferrable {
            sharedBuilder.transferrable = tr
        } else {
            sharedBuilder.particlesCount = Int(b.params.particlesCount)
            sharedBuilder.ug = b.ug
        }
        sharedBuilder.kernelNames = [
            MpmNialltlImpl.kP2g1Kernel,
            MpmNialltlImpl.kP2g2Kernel,
            MpmNialltlImpl.kAdvectKernel,
            MpmNialltlImpl.kG2pKernel,
        ]
        self.shared = sharedBuilder.build(device)
    }
    
    func initParticle(i: Int, pos: Float2, vel: Float2) {
        shared.initParticle(i, pos, vel)
    }
    
    func set(gravity: Float2) {
        shared.set(gravity: gravity)
    }
    
    func getTransferrable() -> Mpm2DTransferrableData {
        return Mpm2DTransferrableData(shared)
    }
    
    func update(_ commandBuffer: MTLCommandBuffer) {
        for _ in 0..<itersCount {
            shared.clearGrid(commandBuffer)
            
            p2g1(commandBuffer)
            p2g2(commandBuffer)
            advect(commandBuffer)
            g2p(commandBuffer)
        }
    }
    
    private func p2g1(_ commandBuffer: MTLCommandBuffer) {
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        let pipelineState = shared.getKernel(MpmNialltlImpl.kP2g1Kernel)
        commandEncoder.setComputePipelineState(pipelineState)
        let buffers = [
            positionsBuffer,
            shared.velocitiesBuffer,
            shared.CsBuffer,
            shared.gridMsBuffer,
            shared.gridVsBuffer,
            shared.ugPack.buffer,
            paramsBuffer,
        ]
        let tgParams = Mtl1DThreadGridParams(threadsPerGrid: particlesCount, threadsPerGroup: pipelineState.maxTotalThreadsPerThreadgroup)
        shared.finish(commandEncoder, buffers, tgParams)
    }

    private func p2g2(_ commandBuffer: MTLCommandBuffer) {
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        let pipelineState = shared.getKernel(MpmNialltlImpl.kP2g2Kernel)
        commandEncoder.setComputePipelineState(pipelineState)
        let buffers = [
            positionsBuffer,
            shared.velocitiesBuffer,
            shared.CsBuffer,
            shared.gridMsBuffer,
            shared.gridVsBuffer,
            shared.ugPack.buffer,
            paramsBuffer,
        ]
        let tgParams = Mtl1DThreadGridParams(threadsPerGrid: particlesCount, threadsPerGroup: pipelineState.maxTotalThreadsPerThreadgroup)
        shared.finish(commandEncoder, buffers, tgParams)
    }
    
    private func advect(_ commandBuffer: MTLCommandBuffer) {
        shared.advect(name: MpmNialltlImpl.kAdvectKernel, paramsBuffer, commandBuffer)
    }
    
    private func g2p(_ commandBuffer: MTLCommandBuffer) {
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        let pipelineState = shared.getKernel(MpmNialltlImpl.kG2pKernel)
        commandEncoder.setComputePipelineState(pipelineState)
        let buffers = [
            positionsBuffer,
            shared.velocitiesBuffer,
            shared.CsBuffer,
            shared.gridMsBuffer,
            shared.gridVsBuffer,
            shared.ugPack.buffer,
            paramsBuffer,
        ]
        let tgParams = Mtl1DThreadGridParams(threadsPerGrid: particlesCount, threadsPerGroup: pipelineState.maxTotalThreadsPerThreadgroup)
        shared.finish(commandEncoder, buffers, tgParams)
    }
}
