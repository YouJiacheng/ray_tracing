import taichi as ti

from utils import ti_dataclass, typing

vec2 = ti.types.vector(2, ti.float32)
vec3 = ti.types.vector(3, ti.float32)

@ti_dataclass
class Ray:
    O: vec3
    D: vec3
RayT = typing(Ray, ti.template())

@ti_dataclass
class RayRecord:
    # convert recursion to iteration using record
    ray: Ray
    accumulated_attenuation: vec3
RayRecordT = typing(RayRecord, ti.template())

@ti_dataclass
class Triangle:
    V0: vec3
    V1: vec3
    V2: vec3
TriangleT = typing(Triangle, ti.template())

@ti_dataclass
class TextureTriangle:
    V0: vec2
    V1: vec2
    V2: vec2
TextureTriangleT = typing(TextureTriangle, ti.template())

@ti_dataclass
class Primitive:
    position_triangle: Triangle
    texture_triangle: TextureTriangle
    material_index: ti.i32
    texture_index: ti.i32
PrimitiveT = typing(Primitive, ti.template())

@ti_dataclass
class Material:
    material_type: ti.i32
    attenuation: vec3
    emissivity: vec3
    factor: ti.f32
