# taichi is incompatible with from __future__ import annotations
from typing import Iterable

import numpy as np
import taichi as ti
import taichi.math as tm

from bvh import BVHNode, BVHNodeStruct, flatten_bvh, triangle_AABB
from camera import Camera, image_height, image_width, make_camera
from structs import (Material, Primitive, PrimitiveT, Ray, RayRecord,
                     RayRecordT, RayT, TextureTriangleT, TriangleT, vec2, vec3)

T_MIN = 1e-4 # fix shadow acne problem

material_map = {
    'nonrefractive': 0,
    'refractive': 1,
}

def parse_material(material_type: str, data_str: str):
    items = data_str.split(',')
    type_index = np.int32(material_map[material_type])
    attenuation = np.array([float(v) / 255 for v in items[:3]], np.float32)
    emissivity = np.array([float(v) / 255 for v in items[3:6]], np.float32)
    match material_type:
        case 'nonrefractive':
            factor = np.float32(items[6]) # lambertian factor
        case 'refractive':
            factor = np.float32(items[6]) # eta
        case _ as mt:
            raise ValueError(f'Unsupported Material Type: {mt}')
    return type_index, attenuation, emissivity, factor

@ti.data_oriented
class Scene:
    def __init__(self):
        import json
        from pathlib import Path
        parent = Path(__file__).parent
        with (parent / 'scene.json').open() as f:
            objects = json.load(f)
        
        material_list = []
        texture_list = []
        def parse(obj: dict[str, str]):
            from pywavefront import Wavefront
            from pywavefront.material import Material as WFMat

            def parse_csv(csv: str):
                return np.array([float(v) for v in csv.split(',')], np.float32)
            scale = parse_csv(obj['scale'])
            offset = parse_csv(obj['offset'])

            path = (parent / obj['mesh']).resolve()
            materials: dict[str, WFMat] = Wavefront(str(path)).materials
            _, material = materials.popitem()
            assert not materials # only contains one
            data = np.array(material.vertices, np.float32)
            match material.vertex_format:
                case 'T2F_V3F':
                    data = data.reshape((-1, 3, 5))
                    postion = data[..., 2:] * scale + offset
                    texture = data[..., :2]
                case 'V3F':
                    postion = data.reshape((-1, 3, 3)) * scale + offset
                    texture = np.zeros_like(postion[..., :2])
                case _ as vf:
                    raise ValueError(f'Unsupported Vertex Format: {vf}')
            v0, v1, v2 = postion.swapaxes(0, 1)
            vt0, vt1, vt2 = texture.swapaxes(0, 1)
            n = len(v0)
            material_index_arr = np.full(shape=n, fill_value=len(material_list), dtype=np.int32)
            material_type, data_str =  obj['material'].split('_')
            material_list.append(parse_material(material_type, data_str))
            texture_index = -1
            if (texture_file := obj.get('texture')) is not None:
                texture_path = (parent / texture_file).resolve()
                texture_full = ti.tools.imread(str(texture_path))
                texture_index = len(texture_list)
                texture_list.append(ti.tools.imresize(texture_full, 512, 512))
            
            texture_index_arr = np.full(shape=n, fill_value=texture_index, dtype=np.int32)

            return v0, v1, v2, vt0, vt1, vt2, material_index_arr, texture_index_arr
        
        def concat(row_chunks: Iterable[tuple]):
            return tuple(np.concatenate(col_chunks) for col_chunks in zip(*row_chunks))
        v0, v1, v2, vt0, vt1, vt2, material_index, texture_index = concat(parse(obj) for obj in objects)
        n = len(v0)
        self.primitives = Primitive.field(shape=n)
        self.primitives.from_numpy(dict(
            position_triangle=dict(
                V0=v0,
                V1=v1,
                V2=v2,
            ),
            texture_triangle=dict(
                V0=vt0,
                V1=vt1,
                V2=vt2,
            ),
            material_index=material_index,
            texture_index=texture_index,
        ))

        def stack(rows: Iterable[tuple]):
            return tuple(np.stack(col, 0) for col in zip(*rows))
        m = len(material_list)
        self.materials = Material.field(shape=m)
        self.materials.from_numpy(dict(zip(Material.members, stack(material_list))))

        n_texts = len(texture_list)
        self.textures: ti.MatrixField = ti.Vector.field(
            n=3, dtype=ti.uint8, shape=(max(n_texts, 1), 512, 512)
        )
        if n_texts >= 1:
            self.textures.from_numpy(np.stack(texture_list, 0))

        aabbs = [triangle_AABB(np.stack([v0[i], v1[i], v2[i]], 0)) for i in range(n)]
        bvh_root = BVHNode(list(range(n)), aabbs)
        self.bvh_arr = BVHNodeStruct.field(shape=bvh_root.n_nodes)
        self.bvh_arr.from_numpy(flatten_bvh(bvh_root))
        print(bvh_root.height)
        print(len(list(bvh_root.walk())))
        print(bvh_root.n_nodes)
        for node in bvh_root.walk():
            if node.primitive_index == -1:
                assert node.left is not None and node.right is not None
                assert node.left.index == node.index + 1, f'{node.left.index} {node.index}'
    
    @ti.func
    def hit_all(self, ray: RayT):
        index = -1
        t_hit = tm.inf
        normal_hit = vec3(0, 0, 0)
        text_uv_hit = vec2(0, 0)
        ti.loop_config(serialize=True)
        for i in range(self.primitives.shape[0]):
            hit, t, normal, text_uv = hit_primitive(ray, self.primitives[i], T_MIN, t_hit)
            if hit:
                index = i
                t_hit = t
                normal_hit = normal
                text_uv_hit = text_uv
        return index, ray.O + t_hit * ray.D, normal_hit, text_uv_hit
    
    @ti.func
    def hit_all_bvh(self, ray: RayT):
        index = -1
        t_hit = tm.inf
        normal_hit = vec3(0, 0, 0)
        text_uv_hit = vec2(0, 0)

        current = 0 # root
        while current != -1:
            node = self.bvh_arr[current]
            i = node.primitive_index
            if i != -1:
                # leaf
                hit, t, normal, text_uv = hit_primitive(ray, self.primitives[i], T_MIN, t_hit)
                if hit:
                    index = i
                    t_hit = t
                    normal_hit = normal
                    text_uv_hit = text_uv
                current = node.next # no child, visit next (sibling or ancestor's sibling)
            elif hit_aabb(ray, node.min, node.max, T_MIN, t_hit):
                # non leaf, must have left and right children
                current += 1 # visit left child
            else:
                current = node.next # child won't be hit, visit next (sibling or ancestor's sibling)

        return index, ray.O + t_hit * ray.D, normal_hit, text_uv_hit

@ti.func
def hit_aabb(ray: RayT, aabb_min: vec3, aabb_max: vec3, t_min: ti.f32, t_max: ti.f32):
    hit = True
    for i in ti.static(range(3)):
        aabb_min_t = (aabb_min[i] - ray.O[i]) / ray.D[i]
        aabb_max_t = (aabb_max[i] - ray.O[i]) / ray.D[i]
        inside_t_min = ti.min(aabb_min_t, aabb_max_t)
        inside_t_max = ti.max(aabb_min_t, aabb_max_t)
        if inside_t_min > t_max or inside_t_max < t_min:
            hit = False
    return hit

@ti.func
def hit_primitive(ray: RayT, primitive: PrimitiveT, t_min: ti.f32, t_max: ti.f32):
    hit, t, u, v, w = hit_triangle(ray, primitive.position_triangle, t_min, t_max)
    normal = triangle_normal(primitive.position_triangle)
    text_uv = triangle_uv(primitive.texture_triangle, u, v, w)
    return hit, t, normal, text_uv


@ti.func
def triangle_uv(tri: TextureTriangleT, u, v, w):
    return u * tri.V0 + v * tri.V1 + w * tri.V2

@ti.func
def triangle_normal(tri: TriangleT):
    return tm.cross(tri.V1 - tri.V0, tri.V2 - tri.V0)

@ti.func
def hit_triangle(ray: RayT, tri: TriangleT, t_min: ti.f32, t_max: ti.f32):
    # uV0 + vV1 + wV2 = O + tD, u = 1 - v - w
    # t(-D) + v(V1 - V0) + w(V2 - V0) = O - V0
    # A (t, v, w) = O - V0
    A = ti.Matrix.cols([-ray.D, tri.V1 - tri.V0, tri.V2 - tri.V0])
    b = ray.O - tri.V0
    t, v, w = tm.nan, tm.nan, tm.nan
    if abs(tm.determinant(A)) > 1e-6:
        t, v, w = ti.solve(A, b, ti.float32)
    u = 1 - v - w
    inside = 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1
    return (inside and t_min <= t <= t_max), t, u, v, w


ti.init(arch=ti.gpu, debug=False)
scene = Scene()
camera = Camera.field(shape=())
pixels_linear: ti.MatrixField = ti.Vector.field(3, ti.f32, shape=(image_width, image_height))
pixels_corrected: ti.MatrixField = ti.Vector.field(3, ti.f32, shape=(image_width, image_height))
ray_records = RayRecord.field(shape=(image_width, image_height))

# time to live(#scatter)
ray_TTLs: ti.ScalarField = ti.field(ti.i32, shape=(image_width, image_height))
pixels_num_samples: ti.ScalarField = ti.field(ti.i32, shape=(image_width, image_height))


@ti.kernel
def clear():
    pixels_linear.fill(0)
    pixels_num_samples.fill(0)
    ray_TTLs.fill(0)

@ti.func
def random_sphere():
    x = ti.randn(ti.float32)
    y = ti.randn(ti.float32)
    z = ti.randn(ti.float32)
    return tm.normalize(vec3(x, y, z))

@ti.kernel
def step():
    max_depth = 16
    max_sample_per_pixel = 512
    for x, y in pixels_linear:
        if pixels_num_samples[x, y] >= max_sample_per_pixel:
            continue
        if ray_TTLs[x, y] <= 0:
            # generate a new ray
            u = (x + ti.random()) / image_width
            v = (y + ti.random()) / image_height
            ray_records[x, y] = RayRecord(
                ray=camera[None].make_ray(u, v),
                accumulated_attenuation=vec3(1, 1, 1),
            )
            ray_TTLs[x, y] = max_depth
        
        received, hit, record = ray_step(ray_records[x, y])
        pixels_linear[x, y] += received
        
        if hit:
            ray_records[x, y] = record
            ray_TTLs[x, y] -= 1
        else:
            ray_TTLs[x, y] = 0 # this ray is dead
        
        if ray_TTLs[x, y] <= 0:
            # complete a sample
            pixels_num_samples[x, y] += 1

@ti.func
def sample_texture(i: ti.i32, uv: vec2):
    x = ti.floor(uv[0] * 512, ti.i32)
    y = ti.floor(uv[1] * 512, ti.i32)
    return ti.cast(scene.textures[i, x, y], ti.float32) / 255

@ti.func
def square(x):
    return x * x

@ti.func
def pow5(x):
    x_square = square(x)
    return x_square * x_square * x

@ti.func
def ray_step(record: RayRecordT):
    received = vec3(0, 0, 0)
    hit = False

    ray = record.ray
    i, pos, normal, uv = scene.hit_all_bvh(ray)
    if i != -1:
        out = vec3(0, 0, 0)
        hit = True
        acc_attenuation = record.accumulated_attenuation
        primitive = scene.primitives[i]
        material = scene.materials[primitive.material_index]
        received = acc_attenuation * material.emissivity
        attenuation = material.attenuation
        # F0 for nonrefractive
        # transmissivity for refractive
        if primitive.texture_index != -1:
            attenuation *= sample_texture(primitive.texture_index, uv)
        d = tm.normalize(ray.D)
        n = tm.normalize(normal)
        k = tm.dot(d, n) # maybe negative
        s = tm.sign(k)
        reflect = d - 2 * k * n # reflect direction(unit norm)
        if material.material_type == 0:
            lambertian = material.factor # lambertian factor
            lambert = tm.normalize(-s * n + random_sphere())
            # reflect direction interpolated between fresnel and lambertian reflection
            out = tm.normalize((1 - lambertian) * reflect + lambertian * lambert)
            half_way = tm.normalize(d - out)
            # schlick approximation
            # reflectivity interpolated between fresnel and lambertian reflection
            f0 = attenuation
            cos_i = tm.dot(d, half_way)
            attenuation = f0 + (1 - lambertian) * (1 - f0) * pow5(1 - cos_i)
            
        elif material.material_type == 1:
            eta = material.factor
            if s < 0: # out -> in
                eta = 1 / eta
            f0 = square((eta - 1) / (eta + 1))
            
            cos_i = abs(k)
            # if 1 - square(eta) * (1 - square(k)) < 0
            # since fresnel factor = 1 when cos_t = 0
            # total reflection is automatically satisfied
            cos_t = tm.sqrt(tm.max(0, 1 - square(eta) * (1 - square(k))))
            cos_min = tm.min(cos_i, cos_t)
            reflectivity = f0 + (1 - f0) * pow5(1 - cos_min)
            if ti.random() <= reflectivity: # reflect
                attenuation = vec3(1, 1, 1)
                out = reflect
            else: # refract
                # attenuation caused by absorption
                # out = tm.refract(d, -s * n, eta)
                out = eta * (d - k * n) + cos_t * s * n
                v = (out - tm.refract(d, -s * n, eta)).norm()
                if v > 1e-3:
                    print(v)

        record = RayRecord(
            ray=Ray(O=pos, D=out),
            accumulated_attenuation=acc_attenuation * attenuation
        )
    return received, hit, record

@ti.kernel
def gamma_correction():
    for x, y in pixels_linear:
        pixels_corrected[x, y] = tm.sqrt(pixels_linear[x, y] / (pixels_num_samples[x, y] + 1))

@ti.kernel
def count_min_samples() -> ti.i32:
    n = 1000000
    for x, y in pixels_num_samples:
        ti.atomic_min(n, pixels_num_samples[x, y])
    return n

@ti.kernel
def count_max_samples() -> ti.i32:
    n = 0
    for x, y in pixels_num_samples:
        ti.atomic_max(n, pixels_num_samples[x, y])
    return n

window = ti.ui.Window('光线追踪', res = (image_width, image_height), pos = (150, 150))
canvas = window.get_canvas()
ui_camera = ti.ui.Camera()
ui_camera.position(-0.5, 0.5, -3)
ui_camera.lookat(0, 0, 0)
from time import time

def make_edge_trigger(init: np.ndarray):
    last_val = init
    def detect(curr_val: np.ndarray):
        nonlocal last_val
        edge = np.linalg.norm(curr_val - last_val, 2) > 1e-3
        last_val = curr_val
        return bool(edge)
    return detect

position_edge = make_edge_trigger(np.array([0, 0, 0]))
lookat_edge = make_edge_trigger(np.array([0, 0, 0]))
while window.running:
    t = time()
    ui_camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.SPACE)
    curr_p = ui_camera.curr_position
    curr_l = ui_camera.curr_lookat
    if (position_edge(np.asarray(curr_p)) or
        lookat_edge(np.asarray(curr_l))):
        camera[None] = make_camera(
            position=curr_p,
            lookat=curr_l,
            up=[0, 1, 0],
        )
        clear()

    step()
    gamma_correction()
    min_samples = count_min_samples()
    max_samples = count_max_samples()
    canvas.set_image(pixels_corrected)
    window.show()
    print(f'{min_samples=}, {max_samples=}, step_time={time() - t}s')

ti.tools.imwrite(pixels_corrected, 'result.png')