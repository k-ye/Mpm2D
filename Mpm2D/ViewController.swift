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
    fileprivate var mpm: Mpm2D!
    fileprivate var renderer: ParticlesRenderer2D!
    
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
        
        initMpm(boundary: to(view.bounds.size) * 0.1)
        initRenderer()
        
        timer = CADisplayLink(target: self, selector: #selector(renderLoop))
        timer.add(to: .main, forMode: .default)
    }
    
    private func initMpm(boundary: Float2) {
        let cellSize: Float = 1.0
        let ugParams = UniformGrid2DParams(boundary: boundary, cellSize: cellSize)
        uniformGridPack = UniformGrid2DParamsHMPack(ugParams, device)
        
        let particlesCount = 8192
        mpm = Mpm2D.Builder()
            .set(uniformGridPack)
            .set(itersCount: 20)
            .set(particlesCount: particlesCount)
            .set(timestep: 1.0 / 1e3)
            .set(rho: 1.0)
            .set(E: 400.0)
            .build(device)
        
        let sideSize = min(boundary[0], boundary[1]) * 0.4
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
    
    func renderOnce() {
        let drawable = metalLayer.nextDrawable()!
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        mpm.update(commandBuffer)
        renderer.render(drawable, commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
