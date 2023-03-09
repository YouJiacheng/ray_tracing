from typing import Iterator

import numpy as np
import taichi as ti

from structs import vec3
from utils import ti_dataclass, transpose


class AABB:
    def __init__(self, box_min: np.ndarray, box_max: np.ndarray):
        self.min = box_min
        self.max = box_max
        self.centroid = box_min + (box_max - box_min) / 2
    
def triangle_AABB(Vs: np.ndarray):
    return AABB(np.amin(Vs, 0), np.amax(Vs, 0))

def merge_AABB(a: AABB, b: AABB):
    return AABB(np.minimum(a.min, b.min), np.maximum(a.max, b.max))

# EqualCounts BVH
class BVHNode:
    primitive_index = -1
    left = None
    right = None
    index = -1
    def __init__(
        self,
        primitive_indices: list[int],
        aabbs: list[AABB],
        parent: 'BVHNode' = None,
    ):
        self.parent = parent
        n_prims = len(primitive_indices)
        assert n_prims >= 1
        
        if n_prims == 1:
            # leaf
            self.primitive_index = primitive_indices[0]
            self.aabb = aabbs[0]
            self.n_nodes = 1
            self.height = 1
        else:
            centroid_max = np.amax([aabb.centroid for aabb in aabbs], 0)
            centroid_min = np.amin([aabb.centroid for aabb in aabbs], 0)
            centroid_span = centroid_max - centroid_min
            i = np.argmax(centroid_span)
            sorted_list = sorted([(p, aabb) for p, aabb in zip(primitive_indices, aabbs)], key=lambda t: t[1].centroid[i])
            mid = int(n_prims / 2)
            self.left = l = BVHNode(*transpose(sorted_list[:mid]), parent=self)
            self.right = r = BVHNode(*transpose(sorted_list[mid:]), parent=self)
            self.aabb = merge_AABB(l.aabb, r.aabb)
            self.n_nodes = l.n_nodes + r.n_nodes + 1
            self.height = max(l.height, r.height) + 1

    def next(self):
        node = self
        while True:
            if (parent := node.parent) is None:
                return None
            if parent.right is not node:
                return parent.right
            node = parent
    
    def walk(self) -> Iterator['BVHNode']:
        yield self
        if self.primitive_index == -1:
            assert self.left is not None
            assert self.right is not None
            yield from self.left.walk()
            yield from self.right.walk()


@ti_dataclass
class BVHNodeStruct:
    min: vec3
    max: vec3
    primitive_index: ti.i32
    # for non-leaf, left = current node index + 1
    # and we always visit left first (no need for right)
    next: ti.i32

def flatten_bvh(bvh: BVHNode):
    # pre enumerate nodes
    for i, node in enumerate(bvh.walk()):
        node.index = i

    n = bvh.n_nodes
    min_arr = np.empty((n, 3), np.float32)
    max_arr = np.empty((n, 3), np.float32)
    primitive_index_arr = np.empty(n, np.int32)
    next_arr = np.full(n, -1, np.int32)
    

    for i, node in enumerate(bvh.walk()):
        min_arr[i] = node.aabb.min
        max_arr[i] = node.aabb.max
        primitive_index_arr[i] = node.primitive_index
        
        next = node.next()
        if next is not None:
            next_arr[i] = next.index
    
    return dict(
        min=min_arr,
        max=max_arr,
        primitive_index=primitive_index_arr,
        next=next_arr,
    )