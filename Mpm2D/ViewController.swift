//
//  ViewController.swift
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

import CoreMotion
import Metal
import UIKit

func to(_ sz: CGSize) -> Float2 {
    return Float2(Float(sz.width), Float(sz.height))
}

fileprivate let kParticlesCount = 16384

class ViewController: UIViewController {
    fileprivate var device: MTLDevice!
    fileprivate var commandQueue: MTLCommandQueue!
    fileprivate var metalLayer: CAMetalLayer!
    fileprivate var timer: CADisplayLink!
    
    fileprivate var mpmBoundary: Float2 = .zero
    fileprivate var uniformGridPack: UniformGrid2DParamsHMPack!
    fileprivate var mpm: Mpm2DSolver!
    fileprivate var renderer: ParticlesRenderer2D!
    
    fileprivate var paused = true
    
    let motion = CMMotionManager()
    @IBOutlet var singleTapGesture: UITapGestureRecognizer!
    @IBOutlet var doubleTapGesture: UITapGestureRecognizer!
    @IBOutlet weak var mpmSelector: UISegmentedControl!
    
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
        view.bringSubviewToFront(mpmSelector)
        
        mpmBoundary = to(view.bounds.size) * 0.2
        initUniformGrid()
        initMpmByIndex(mpmSelector.selectedSegmentIndex)
        initPartciles()
        initRenderer()
        
        timer = CADisplayLink(target: self, selector: #selector(renderLoop))
        timer.add(to: .main, forMode: .default)
        
        singleTapGesture.numberOfTapsRequired = 1
        doubleTapGesture.numberOfTapsRequired = 2
        view.addGestureRecognizer(singleTapGesture)
        view.addGestureRecognizer(doubleTapGesture)
        // https://stackoverflow.com/a/8876299/12003165
        singleTapGesture.require(toFail: doubleTapGesture)
        
        // https://developer.apple.com/documentation/coremotion/getting_raw_accelerometer_events
        if motion.isAccelerometerAvailable {
            motion.accelerometerUpdateInterval = 1.0 / 60.0  // 60 Hz
            motion.startAccelerometerUpdates()
        }
        
//        self.becomeFirstResponder() // To get shake gesture
    }
    
    private func initUniformGrid() {
        let cellSize: Float = 1.0
        let ugParams = UniformGrid2DParams(boundary: mpmBoundary, cellSize: cellSize)
        uniformGridPack = UniformGrid2DParamsHMPack(ugParams, device)
        
        let ug = uniformGridPack.params
        print("UniformGrid boundary=\(ug.boundary) grid=\(ug.grid) cellSize=\(cellSize) cellsCount=\(ug.cellsCount)")
    }
    
    private func initMpmByIndex(_ mpmIdx: Int) {
        initMpmByIndex(mpmIdx, transferrable: nil)
    }
    
    private func initMpmByIndex(_ mpmIdx: Int, transferrable: Mpm2DTransferrableData?) {
        let make = { () -> Mpm2DSolver in
            if mpmIdx == 0 {
                return Mpm88SolverBuilder()
                    .set(self.uniformGridPack)
                    .set(itersCount: 5)
                    .set(particlesCount: kParticlesCount)
                    .set(timestep: 10.0 / 1e3)
                    .set(rho: 1.0)
                    .set(E: 400.0)
                    .set(transferrable)
                    .build(self.device)
            }
            return MpmNialltlSolverBuilder()
                .set(self.uniformGridPack)
                .set(itersCount: 2)
                .set(particlesCount: kParticlesCount)
                .set(timestep: 25.0 / 1e3)
                .set(restDensity: 1.0)
                .set(dynamicViscosity: 0.1)
                .set(eosStiffness: 2.0)
                .set(eosPower: 4.0)
                .set(transferrable)
                .build(self.device)
        }
        mpm = make()
    }
    
    private func initPartciles() {
        let sideSize = min(mpmBoundary[0], mpmBoundary[1]) * 0.5
        let offsetX = (mpmBoundary[0] - sideSize) * 0.5
        let offsetY = (mpmBoundary[1] - sideSize) * 0.5
        for i in 0..<kParticlesCount {
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
            .build(device)
    }
    
    @objc func renderLoop() {
        autoreleasepool {
            renderOnce()
        }
    }
    
    private func processAccelerometer() {
        if let data = motion.accelerometerData {
            let x = Float(data.acceleration.x)
            let y = Float(data.acceleration.y)
            mpm.set(gravity: Float2(x, y))
        }
    }
    
    private func renderOnce() {
        let drawable = metalLayer.nextDrawable()!
        let commandBuffer = commandQueue.makeCommandBuffer()!
        if !paused {
            processAccelerometer()
            mpm.update(commandBuffer)
        }
        renderer.render(mpm, drawable, commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    @IBAction func onPickerIndexChanged(_ sender: UISegmentedControl) {
        initMpmByIndex(sender.selectedSegmentIndex, transferrable: mpm.getTransferrable())
    }
    
    @IBAction func handleSingleTap(_ sender: UITapGestureRecognizer) {
        paused = !paused
    }
    
    @IBAction func handleDoubleTap(_ sender: UITapGestureRecognizer) {
        paused = true
        let alert = UIAlertController(title: "Reset?", message: "Shake to reset the particles", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Yes", style: .default, handler: { action in
            self.initMpmByIndex(self.mpmSelector.selectedSegmentIndex)
            self.initPartciles()
        }))
        alert.addAction(UIAlertAction(title: "No", style: .cancel, handler: { action in
            self.paused = false
        }))

        present(alert, animated: true)
    }
}
