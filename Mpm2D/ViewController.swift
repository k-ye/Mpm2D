//
//  ViewController.swift
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

import Metal
import UIKit

func to(_ sz: CGSize) -> Float2 {
    return Float2(Float(sz.width), Float(sz.height))
}

class ViewController: UIViewController {
    fileprivate var device: MTLDevice!
    fileprivate var commandQueue: MTLCommandQueue!
    fileprivate var metalLayer: CAMetalLayer!
    fileprivate var timer: CADisplayLink!
    
    fileprivate var uniformGridPack: UniformGrid2DParamsHMPack!
    fileprivate var mpm: Mpm2DSolver!
    fileprivate var renderer: ParticlesRenderer2D!
    
    fileprivate var paused = true
    @IBOutlet var tapGestureRecognizer: UITapGestureRecognizer!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device.makeCommandQueue()
        
        metalLayer = CAMetalLayer()
        metalLayer.device = device
        // Have to use BGRA
        // https://developer.apple.com/documentation/quartzcore/cametallayer/1478155-pixelformat
        metalLayer.pixelFormat = .bgra8Unorm
        metalLayer.framebufferOnly = true
        metalLayer.frame = view.layer.frame
        view.layer.addSublayer(metalLayer)
        
        let boundary = to(view.bounds.size) * 0.2
        initUniformGrid(boundary: boundary)
        initMpm(boundary: boundary)
        initRenderer()
        
        timer = CADisplayLink(target: self, selector: #selector(renderLoop))
        timer.add(to: .main, forMode: .default)
        
        view.addGestureRecognizer(tapGestureRecognizer)
    }
    
    private func initUniformGrid(boundary: Float2) {
        let cellSize: Float = 1.0
        let ugParams = UniformGrid2DParams(boundary: boundary, cellSize: cellSize)
        uniformGridPack = UniformGrid2DParamsHMPack(ugParams, device)
        
        let ug = uniformGridPack.params
        print("UniformGrid boundary=\(ug.boundary) grid=\(ug.grid) cellSize=\(cellSize) cellsCount=\(ug.cellsCount)")
    }
    
    private func initMpm(boundary: Float2) {
        let particlesCount = 8192
//        mpm = MpmFluidSolverBuilder()
//            .set(uniformGridPack)
//            .set(itersCount: 5)
//            .set(particlesCount: particlesCount)
//            .set(timestep: 10.0 / 1e3)
//            .set(restDensity: 1.0)
//            .set(dynamicViscosity: 0.1)
//            .set(eosStiffness: 10.0)
//            .set(eosPower: 4.0)
//            .build(device)
        mpm = Mpm88SolverBuilder()
            .set(uniformGridPack)
            .set(itersCount: 5)
            .set(particlesCount: particlesCount)
            .set(timestep: 10.0 / 1e3)
            .set(rho: 1.0)
            .set(E: 400.0)
            .build(device)
        
        let sideSize = min(boundary[0], boundary[1]) * 0.5
        let offsetX = (boundary[0] - sideSize) * 0.5
        let offsetY = (boundary[1] - sideSize) * 0.5
        for i in 0..<particlesCount {
            let pos = Float2((Float.random(in: 0..<sideSize) + offsetX),
                             (Float.random(in: 0..<sideSize) + offsetY))
            let vel = Float2(Float.random(in: -0.5..<0.5),
                             Float.random(in: -1.0..<0.0) - 0.5)
            mpm.initParticle(i: i, pos: pos, vel: vel)
        }
    }
    
    private func initRenderer() {
        renderer = ParticlesRenderer2D.Builder()
            .set(uniformGridPack)
            .set(mpm)
            .build(device)
    }
    
    @objc func renderLoop() {
        autoreleasepool {
            renderOnce()
        }
    }
    
    private func renderOnce() {
        let drawable = metalLayer.nextDrawable()!
        let commandBuffer = commandQueue.makeCommandBuffer()!
        if !paused {
            mpm.update(commandBuffer)
        }
        renderer.render(drawable, commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    
    @IBAction func handleTap(_ sender: UITapGestureRecognizer) {
        paused = !paused
    }
}
