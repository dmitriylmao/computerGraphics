import math
from dataclasses import dataclass
import numpy as np
from PIL import Image

def normalize(v): #Нормализация вектора
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        return v
    return v / n

@dataclass
class Material:
    color: np.ndarray  #основной цвет
    diffuse: float     #сила матового отражения от источников
    ambient: float     #фоновое освещение

@dataclass
class Light:
    position: np.ndarray
    color: np.ndarray
    intensity: float 


class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.material = material

    def intersect(self, origin, direction): #есть формула пересечения луча и сферы, 
                                            #решаем квадратное уравнение и берем ближайший 
                                            #положительный корень (o + t d - c)^2 = r^2
                                            #получаю t, если >0 то луч попал
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


class Box: 
    def __init__(self, min_corner, max_corner, material):
        self.min = np.array(min_corner, dtype=np.float32)
        self.max = np.array(max_corner, dtype=np.float32)
        self.material = material

    def intersect(self, origin, direction):
        tmin = -1e9
        tmax = 1e9

        for i in range(3):
            if abs(direction[i]) < 1e-6:
                if origin[i] < self.min[i] or origin[i] > self.max[i]:# луч параллелен; если вне интервала — промах
                    return None
            else:
                t1 = (self.min[i] - origin[i]) / direction[i]
                t2 = (self.max[i] - origin[i]) / direction[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                if t1 > tmin:
                    tmin = t1       #Луч пересекает коробку, если есть общее пересечение всех трёх интервалов
                                    #tmin = max(tx_min, ty_min, tz_min)
                                    #tmax = min(tx_max, ty_max, tz_max)
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

        n = np.zeros(3, dtype=np.float32) # Определяем нормаль по тому, к какой грани ближе всего
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


class PlaneRect: #тсена комнаты
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


def trace_ray(origin, direction, objects): #перебираем все объекты и вызываем intersect длшя каждого
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

#Если по пути до лампы луч врезается хоть во что-то 
#→ свет закрыт → точка в тени от этого источника
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


def shade(point, normal, material, lights, objects): #цвет точки
    color = material.color * material.ambient

    for light in lights:
        to_light = light.position - point #считаем направление и расстояние
        dist = math.sqrt(np.dot(to_light, to_light))
        l_dir = to_light / dist

        ndotl = np.dot(normal, l_dir)
        if ndotl <= 0.0: #сли свет сзади → ноль
            continue

        if is_in_shadow(point, l_dir, dist, objects): #если другой объект закрывает свет → ноль
            continue

        #чем ближе и чем перпендикулярнее — тем ярче
        atten = light.intensity / (dist * dist)
        diffuse = material.diffuse * ndotl * atten
        color += material.color * light.color * diffuse

    return np.clip(color, 0.0, 1.0)


def render():
    width = 320
    height = 240
    aspect = width / float(height)

    fov = math.radians(60.0)
    scale = math.tan(fov * 0.5)

    cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32) # Камера в начале координат

    red_wall   = Material(np.array([0.8, 0.1, 0.1], dtype=np.float32), diffuse=0.7, ambient=0.15)
    blue_wall  = Material(np.array([0.1, 0.3, 0.8], dtype=np.float32), diffuse=0.7, ambient=0.15)
    gray_wall  = Material(np.array([0.75, 0.75, 0.75], dtype=np.float32), diffuse=0.7, ambient=0.2)

    yellow_mat = Material(np.array([0.9, 0.8, 0.1], dtype=np.float32), diffuse=0.8, ambient=0.1)
    green_mat  = Material(np.array([0.1, 0.8, 0.3], dtype=np.float32), diffuse=0.8, ambient=0.1)
    purple_mat = Material(np.array([0.7, 0.3, 0.9], dtype=np.float32), diffuse=0.8, ambient=0.1)
    orange_mat = Material(np.array([0.9, 0.5, 0.1], dtype=np.float32), diffuse=0.8, ambient=0.1)

    objects = [] # Стены комнаты: куб [-1,1] x [-1,1] x [0,4]

    objects.append(PlaneRect(
        axis='x', k=-1.0,
        min1=-1.0, max1=1.0,
        min2=0.0, max2=4.0,
        normal=[1.0, 0.0, 0.0],
        material=red_wall
    ))

    objects.append(PlaneRect(
        axis='x', k=1.0,
        min1=-1.0, max1=1.0,
        min2=0.0, max2=4.0,
        normal=[-1.0, 0.0, 0.0],
        material=blue_wall
    ))

    objects.append(PlaneRect(
        axis='y', k=-1.0,
        min1=-1.0, max1=1.0,
        min2=0.0, max2=4.0,
        normal=[0.0, 1.0, 0.0],
        material=gray_wall
    ))

    objects.append(PlaneRect(
        axis='y', k=1.0,
        min1=-1.0, max1=1.0,
        min2=0.0, max2=4.0,
        normal=[0.0, -1.0, 0.0],
        material=gray_wall
    ))

    objects.append(PlaneRect(
        axis='z', k=4.0,
        min1=-1.0, max1=1.0,
        min2=-1.0, max2=1.0,
        normal=[0.0, 0.0, -1.0],
        material=gray_wall
    ))

    #2 шара + 2 куба
    objects.append(Sphere(center=[-0.4, -0.7, 2.0], radius=0.3, material=yellow_mat))
    objects.append(Sphere(center=[0.5, -0.6, 3.0], radius=0.4, material=green_mat))

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
            intensity=2.0),
        Light(position=np.array([-0.6, 0.7, 3.5], dtype=np.float32),
            color=np.array([1.0, 0.95, 0.9], dtype=np.float32),
            intensity=1.0),
    ]
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

    #Конвертация и сохранение
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    image = Image.fromarray(img_uint8, mode="RGB")
    image.save("cornell.png")
    print("Готово: сохранено в cornell.png")


if __name__ == "__main__":
    render()
