import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


# ---------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------------

def normalize(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        return v
    return v / n


# ---------------- МАТЕРИАЛЫ И СВЕТ ----------------

@dataclass
class Material:
    color: np.ndarray  # базовый цвет (RGB 0..1)
    diffuse: float     # доля диффузного света
    ambient: float     # фон


@dataclass
class Light:
    position: np.ndarray
    color: np.ndarray
    intensity: float   # множитель силы света


# ---------------- ОБЪЕКТЫ СЦЕНЫ ----------------

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.material = material

    def intersect(self, origin, direction):
        # Решаем (o + t d - c)^2 = r^2
        oc = origin - self.center
        b = np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        disc = b * b - c
        if disc < 0.0:
            return None
        sqrt_disc = math.sqrt(disc)

        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc

        t = None
        eps = 1e-4
        if t1 > eps:
            t = t1
        elif t2 > eps:
            t = t2
        else:
            return None

        hit_point = origin + t * direction
        normal = normalize(hit_point - self.center)
        return t, hit_point, normal, self.material


class Box:  # axis-aligned
    def __init__(self, min_corner, max_corner, material):
        self.min = np.array(min_corner, dtype=np.float32)
        self.max = np.array(max_corner, dtype=np.float32)
        self.material = material

    def intersect(self, origin, direction):
        tmin = -1e9
        tmax = 1e9

        for i in range(3):
            if abs(direction[i]) < 1e-6:
                # луч параллелен; если вне интервала — промах
                if origin[i] < self.min[i] or origin[i] > self.max[i]:
                    return None
            else:
                t1 = (self.min[i] - origin[i]) / direction[i]
                t2 = (self.max[i] - origin[i]) / direction[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > tmin:
                    tmin = t1
                if t2 < tmax:
                    tmax = t2
                if tmax < tmin:
                    return None

        eps = 1e-4
        if tmax < eps:
            return None

        t_hit = tmin if tmin > eps else tmax
        if t_hit < eps:
            return None

        hit_point = origin + t_hit * direction

        # Определяем нормаль по тому, к какой грани ближе всего
        n = np.zeros(3, dtype=np.float32)
        if abs(hit_point[0] - self.min[0]) < 1e-3:
            n = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        elif abs(hit_point[0] - self.max[0]) < 1e-3:
            n = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif abs(hit_point[1] - self.min[1]) < 1e-3:
            n = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        elif abs(hit_point[1] - self.max[1]) < 1e-3:
            n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif abs(hit_point[2] - self.min[2]) < 1e-3:
            n = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return t_hit, hit_point, n, self.material


class PlaneRect:
    """
    Ограниченная плоскость (стена комнаты).
    axis: 'x', 'y' или 'z'
    k: координата по этой оси
    min1..max1, min2..max2: диапазоны по двум другим осям
    normal: внутренняя нормаль
    """
    def __init__(self, axis, k, min1, max1, min2, max2, normal, material):
        self.axis = axis
        self.k = float(k)
        self.min1 = float(min1)
        self.max1 = float(max1)
        self.min2 = float(min2)
        self.max2 = float(max2)
        self.normal = normalize(np.array(normal, dtype=np.float32))
        self.material = material

    def intersect(self, origin, direction):
        eps = 1e-4

        if self.axis == 'x':
            if abs(direction[0]) < 1e-6:
                return None
            t = (self.k - origin[0]) / direction[0]
            if t <= eps:
                return None
            hit = origin + t * direction
            y, z = hit[1], hit[2]
            if self.min1 <= y <= self.max1 and self.min2 <= z <= self.max2:
                return t, hit, self.normal, self.material
            return None

        if self.axis == 'y':
            if abs(direction[1]) < 1e-6:
                return None
            t = (self.k - origin[1]) / direction[1]
            if t <= eps:
                return None
            hit = origin + t * direction
            x, z = hit[0], hit[2]
            if self.min1 <= x <= self.max1 and self.min2 <= z <= self.max2:
                return t, hit, self.normal, self.material
            return None

        if self.axis == 'z':
            if abs(direction[2]) < 1e-6:
                return None
            t = (self.k - origin[2]) / direction[2]
            if t <= eps:
                return None
            hit = origin + t * direction
            x, y = hit[0], hit[1]
            if self.min1 <= x <= self.max1 and self.min2 <= y <= self.max2:
                return t, hit, self.normal, self.material
            return None

        return None


# ---------------- ТРАССИРОВКА ЛУЧА ----------------

def trace_ray(origin, direction, objects):
    closest_t = 1e9
    hit_point = None
    hit_normal = None
    hit_material = None

    for obj in objects:
        res = obj.intersect(origin, direction)
        if res is None:
            continue
        t, p, n, m = res
        if t < closest_t:
            closest_t = t
            hit_point = p
            hit_normal = n
            hit_material = m

    if hit_point is None:
        return None
    return hit_point, hit_normal, hit_material


def is_in_shadow(point, light_dir, max_dist, objects):
    shadow_origin = point + light_dir * 1e-4
    for obj in objects:
        res = obj.intersect(shadow_origin, light_dir)
        if res is None:
            continue
        t, _, _, _ = res
        if 1e-4 < t < max_dist:
            return True
    return False


def shade(point, normal, material, lights, objects):
    color = material.color * material.ambient

    for light in lights:
        to_light = light.position - point
        dist = math.sqrt(np.dot(to_light, to_light))
        l_dir = to_light / dist

        # косинус угла между нормалью и направлением на свет
        ndotl = np.dot(normal, l_dir)
        if ndotl <= 0.0:
            continue

        # тень
        if is_in_shadow(point, l_dir, dist, objects):
            continue

        # диффузное освещение, ослабление по расстоянию
        atten = light.intensity / (dist * dist)
        diffuse = material.diffuse * ndotl * atten
        color += material.color * light.color * diffuse

    # ограничиваем в [0,1]
    return np.clip(color, 0.0, 1.0)


# ---------------- СБОРКА СЦЕНЫ И РЕНДЕР ----------------

def render():
    # Размер изображения
    width = 320
    height = 240
    aspect = width / float(height)

    fov = math.radians(60.0)
    scale = math.tan(fov * 0.5)

    # Камера в начале координат, смотрит вдоль +Z
    cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Материалы
    red_wall = Material(np.array([0.8, 0.1, 0.1], dtype=np.float32), diffuse=0.9, ambient=0.05)
    blue_wall = Material(np.array([0.1, 0.3, 0.8], dtype=np.float32), diffuse=0.9, ambient=0.05)
    gray_wall = Material(np.array([0.75, 0.75, 0.75], dtype=np.float32), diffuse=0.9, ambient=0.05)

    yellow_mat = Material(np.array([0.9, 0.8, 0.1], dtype=np.float32), diffuse=0.9, ambient=0.05)
    green_mat = Material(np.array([0.1, 0.8, 0.3], dtype=np.float32), diffuse=0.9, ambient=0.05)
    purple_mat = Material(np.array([0.7, 0.3, 0.9], dtype=np.float32), diffuse=0.9, ambient=0.05)
    orange_mat = Material(np.array([0.9, 0.5, 0.1], dtype=np.float32), diffuse=0.9, ambient=0.05)

    # Стены комнаты: куб [-1,1] x [-1,1] x [0,4]
    objects = []

    # левая стена (красная)
    objects.append(PlaneRect(
        axis='x', k=-1.0,
        min1=-1.0, max1=1.0,
        min2=0.0, max2=4.0,
        normal=[1.0, 0.0, 0.0],
        material=red_wall
    ))

    # правая стена (синяя)
    objects.append(PlaneRect(
        axis='x', k=1.0,
        min1=-1.0, max1=1.0,
        min2=0.0, max2=4.0,
        normal=[-1.0, 0.0, 0.0],
        material=blue_wall
    ))

    # пол
    objects.append(PlaneRect(
        axis='y', k=-1.0,
        min1=-1.0, max1=1.0,
        min2=0.0, max2=4.0,
        normal=[0.0, 1.0, 0.0],
        material=gray_wall
    ))

    # потолок
    objects.append(PlaneRect(
        axis='y', k=1.0,
        min1=-1.0, max1=1.0,
        min2=0.0, max2=4.0,
        normal=[0.0, -1.0, 0.0],
        material=gray_wall
    ))

    # задняя стена
    objects.append(PlaneRect(
        axis='z', k=4.0,
        min1=-1.0, max1=1.0,
        min2=-1.0, max2=1.0,
        normal=[0.0, 0.0, -1.0],
        material=gray_wall
    ))

    # Объекты внутри: 2 шара + 2 куба
    objects.append(Sphere(center=[-0.4, -0.3, 2.0], radius=0.3, material=yellow_mat))
    objects.append(Sphere(center=[0.5, -0.2, 3.2], radius=0.4, material=green_mat))

    objects.append(Box(min_corner=[-0.2, -1.0, 1.0],
                       max_corner=[0.2, -0.3, 1.6],
                       material=purple_mat))

    objects.append(Box(min_corner=[-0.8, -1.0, 2.5],
                       max_corner=[-0.3, 0.0, 3.0],
                       material=orange_mat))

    # Источники света
    lights = [
        Light(position=np.array([0.0, 0.9, 1.5], dtype=np.float32),
              color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
              intensity=12.0),
        Light(position=np.array([-0.6, 0.7, 3.5], dtype=np.float32),
              color=np.array([1.0, 0.95, 0.9], dtype=np.float32),
              intensity=5.0),
    ]

    # Буфер изображения (float 0..1)
    img = np.zeros((height, width, 3), dtype=np.float32)

    # Трассировка
    for y in range(height):
        ndc_y = 1.0 - 2.0 * (y + 0.5) / float(height)  # от +1 (верх) до -1 (низ)
        for x in range(width):
            ndc_x = 2.0 * (x + 0.5) / float(width) - 1.0  # от -1 до +1

            px = ndc_x * aspect * scale
            py = ndc_y * scale
            direction = normalize(np.array([px, py, 1.0], dtype=np.float32))

            hit = trace_ray(cam_pos, direction, objects)
            if hit is None:
                # фон — чёрный
                continue

            point, normal, material = hit
            color = shade(point, normal, material, lights, objects)
            img[y, x, :] = color

    # Конвертация в uint8 и сохранение
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    image = Image.fromarray(img_uint8, mode="RGB")
    image.save("cornell.png")
    print("Готово: сохранено в cornell.png")


if __name__ == "__main__":
    render()
