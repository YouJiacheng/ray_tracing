import numpy as np
import taichi as ti

from structs import Ray, vec3
from utils import ti_dataclass, typing

fov = 60.0
image_width = 640
image_height = 640
aspect_ratio = image_width / image_height

@ti_dataclass
class Camera:
    o: vec3
    x: vec3
    y: vec3
    z: vec3
    @ti.func
    def make_ray(self, u: ti.f32, v: ti.f32):
        # +y is up, +z is front
        # thus +x is left
        # but +u is right
        return Ray(
            O=self.o,
            D=(self.x * (0.5 - u) + self.y * (v - 0.5) + self.z),
        )
CameraT = typing(Camera, Camera)


def make_camera(position, lookat, up) -> CameraT:
    pos = np.asarray(position, np.float32)
    at = np.asarray(lookat, np.float32)
    u = np.asarray(up, np.float32)
    def normalize(v: np.ndarray):
        return v / np.linalg.norm(v, 2)
    z = normalize(at - pos)
    x = normalize(np.cross(u, z))
    y = normalize(np.cross(z, x))
    s = 2 * np.tan(np.radians(fov / 2))
    return Camera(o=vec3(pos), x=vec3(s * x * aspect_ratio), y=vec3(s * y), z=vec3(z))