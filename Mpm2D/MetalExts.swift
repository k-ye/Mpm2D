//
//  MetalExts.swift
//  Mpm2D
//
//  Created by Ye Kuang on 2020/07/25.
//  Copyright © 2020 zkk. All rights reserved.
//

import Metal

extension MTLBuffer {
    func toPtr<T>(type: T.Type, count: Int) -> UnsafeBufferPointer<T> {
        return UnsafeBufferPointer(start: contents().bindMemory(to: type, capacity: count), count: count)
    }

    func toMutablePtr<T>(type: T.Type, count: Int) -> UnsafeMutableBufferPointer<T> {
        return UnsafeMutableBufferPointer(start: contents().bindMemory(to: type, capacity: count), count: count)
    }
    
    func toMutablePtr<T>(type: T.Type) -> UnsafeMutablePointer<T> {
        return contents().bindMemory(to: type, capacity: 1)
    }
    
    func toArray<T>(type: T.Type, count: Int) -> [T] {
        var result = [T]()
        toPtr(type: type, count: count).forEach({i in result.append(i)})
        return result
    }
}
