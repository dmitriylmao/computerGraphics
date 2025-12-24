# py -m pip install glfw PyOpenGL PyOpenGL_accelerate numpy pillow

import sys
import ctypes

import glfw
import numpy as np
from OpenGL.GL import *
from PIL import Image

# ============================
# Ассеты
# ============================
OBJ_PATH = "model2.obj"
TEX_PATH = "texture.png"

# Подстройка размера/поворота модели (у OBJ разные масштабы)
MODEL_SCALE = 1.0
MODEL_ROT_SPEED = 0.6  # рад/сек

# ============================
# GLSL: Phong + точечный источник
# ============================
VERTEX_SHADER_SRC = """
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTex;
layout(location = 2) in vec3 aNormal;

uniform struct PointLight {
    vec4 position;      // w=1 для точечного источника
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    vec3 attenuation;   // (k0, k1, k2)
} light;

uniform struct Transform {
    mat4 model;
    mat4 viewProjection;
    mat3 normal;
    vec3 viewPosition;
} transform;

out Vertex {
    vec2 texcoord;
    vec3 normal;
    vec3 lightDir;
    vec3 viewDir;
    float distance;
} Vert;

void main(void)
{
    vec4 vertex = transform.model * vec4(aPos, 1.0);
    vec4 lightDir = light.position - vertex;

    gl_Position = transform.viewProjection * vertex;

    Vert.texcoord = aTex;
    Vert.normal = transform.normal * aNormal;
    Vert.lightDir = vec3(lightDir);
    Vert.viewDir = transform.viewPosition - vec3(vertex);
    Vert.distance = length(lightDir);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core

layout(location = 0) out vec4 color;

uniform struct PointLight {
    vec4 position;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    vec3 attenuation;
} light;

uniform struct Material {
    sampler2D texture;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    vec4 emission;
    float shininess;
} material;

in Vertex {
    vec2 texcoord;
    vec3 normal;
    vec3 lightDir;
    vec3 viewDir;
    float distance;
} Vert;

void main(void)
{
    vec3 normal = normalize(Vert.normal);
    vec3 lightDir = normalize(Vert.lightDir);
    vec3 viewDir = normalize(Vert.viewDir);

    float d = Vert.distance;
    float attenuation = 1.0 / (light.attenuation.x + light.attenuation.y * d + light.attenuation.z * d * d);

    vec4 outColor = material.emission;

    // ambient
    outColor += material.ambient * light.ambient * attenuation;

    // diffuse
    float NdotL = max(dot(normal, lightDir), 0.0);
    outColor += material.diffuse * light.diffuse * NdotL * attenuation;

    // specular
    if (NdotL > 0.0) {
        float RdotV = max(dot(reflect(-lightDir, normal), viewDir), 0.0);
        float spec = pow(RdotV, material.shininess);
        outColor += material.specular * light.specular * spec * attenuation;
    }

    // текстура как базовый цвет
    outColor *= texture(material.texture, Vert.texcoord);

    color = outColor;
}
"""


# ============================
# Math
# ============================
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-8:
        return v
    return v / n


def perspective(fovy_rad: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    f = 1.0 / np.tan(fovy_rad / 2.0)
    a = (z_far + z_near) / (z_near - z_far)
    b = (2.0 * z_far * z_near) / (z_near - z_far)

    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, a, b],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype=np.float32)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _normalize(target - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f

    t = np.eye(4, dtype=np.float32)
    t[0, 3] = -eye[0]
    t[1, 3] = -eye[1]
    t[2, 3] = -eye[2]

    return m @ t


def translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotate_y(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)


# ============================
# OpenGL helpers
# ============================
def compile_shader(source: str, shader_type: int) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    ok = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not ok:
        info = glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ошибка компиляции шейдера:\n{info}")

    return shader


def create_program(vs_src: str, fs_src: str) -> int:
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)

    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)

    ok = glGetProgramiv(prog, GL_LINK_STATUS)
    if not ok:
        info = glGetProgramInfoLog(prog).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ошибка линковки программы:\n{info}")

    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


def uniform_loc(program: int, name: str) -> int:
    loc = glGetUniformLocation(program, name)
    if loc < 0:
        # допускаем -1 (например, если оптимизатор GLSL выкинул неиспользуемое)
        return -1
    return loc


# ============================
# OBJ loader: позиции + UV + НОРМАЛИ (vn)
# ============================
def load_obj(path: str):
    positions = []
    texcoords = []
    normals = []

    vertex_data = []  # [px,py,pz, u,v, nx,ny,nz ...]
    indices = []
    vert_map = {}  # (vi, ti, ni) -> new_index

    def fix_index(i: int, length: int) -> int:
        # OBJ: 1-based; отрицательные — от конца
        return (length + i) if i < 0 else (i - 1)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue

            tag = parts[0]

            if tag == "v" and len(parts) >= 4:
                positions.append(tuple(map(float, parts[1:4])))

            elif tag == "vt" and len(parts) >= 3:
                u, v = map(float, parts[1:3])
                texcoords.append((u, v))

            elif tag == "vn" and len(parts) >= 4:
                normals.append(tuple(map(float, parts[1:4])))

            elif tag == "f" and len(parts) >= 4:
                face = []

                for tok in parts[1:]:
                    # варианты: v/vt/vn, v//vn, v/vt, v
                    vals = tok.split("/")

                    vi = int(vals[0]) if vals[0] else 0
                    ti = int(vals[1]) if len(vals) > 1 and vals[1] else 0
                    ni = int(vals[2]) if len(vals) > 2 and vals[2] else 0

                    vi = fix_index(vi, len(positions)) if vi != 0 else 0
                    ti = fix_index(ti, len(texcoords)) if ti != 0 else -1
                    ni = fix_index(ni, len(normals)) if ni != 0 else -1

                    key = (vi, ti, ni)
                    if key not in vert_map:
                        px, py, pz = positions[vi]

                        if ti >= 0 and ti < len(texcoords):
                            tu, tv = texcoords[ti]
                        else:
                            tu, tv = 0.0, 0.0

                        if ni >= 0 and ni < len(normals):
                            nx, ny, nz = normals[ni]
                        else:
                            # если в OBJ нет нормалей, оставим заглушку
                            nx, ny, nz = 0.0, 0.0, 1.0

                        new_index = len(vertex_data) // 8
                        vert_map[key] = new_index
                        vertex_data.extend([px, py, pz, tu, tv, nx, ny, nz])

                    face.append(vert_map[key])

                # triangulation fan
                for i in range(1, len(face) - 1):
                    indices.extend([face[0], face[i], face[i + 1]])

    if not vertex_data or not indices:
        raise RuntimeError(f"OBJ не загрузился или пустой: {path}")

    vertices_np = np.array(vertex_data, dtype=np.float32)
    indices_np = np.array(indices, dtype=np.uint32)
    return vertices_np, indices_np


# ============================
# Texture loader
# ============================
def load_texture(path: str) -> int:
    img = Image.open(path).convert("RGBA")
    data = np.array(img)[::-1, :, :]
    h, w, _ = data.shape

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        w,
        h,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        data,
    )

    glGenerateMipmap(GL_TEXTURE_2D)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


# ============================
# App
# ============================
def main() -> int:
    if not glfw.init():
        print("GLFW не инициализировался", file=sys.stderr)
        return 1

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # для macOS
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(1280, 720, "Lab14: PointLight + Phong", None, None)
    if not window:
        glfw.terminate()
        print("Не удалось создать окно", file=sys.stderr)
        return 1

    glfw.make_context_current(window)

    def on_resize(_w, width, height):
        glViewport(0, 0, width, height)

    glfw.set_framebuffer_size_callback(window, on_resize)

    # GL state
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    # assets
    vertices, indices = load_obj(OBJ_PATH)
    texture_id = load_texture(TEX_PATH)

    # program
    program = create_program(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)

    # VAO/VBO/EBO
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    stride = 8 * vertices.itemsize  # pos3 + uv2 + normal3

    # aPos
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

    # aTex
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * vertices.itemsize))

    # aNormal
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(5 * vertices.itemsize))

    glBindVertexArray(0)

    # uniforms (часть задаём 1 раз)
    glUseProgram(program)

    # текстура
    tex_loc = uniform_loc(program, "material.texture")
    if tex_loc >= 0:
        glUniform1i(tex_loc, 0)

    # material
    def set_vec4(name, v):
        loc = uniform_loc(program, name)
        if loc >= 0:
            glUniform4f(loc, float(v[0]), float(v[1]), float(v[2]), float(v[3]))

    def set_vec3(name, v):
        loc = uniform_loc(program, name)
        if loc >= 0:
            glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))

    def set_f(name, x):
        loc = uniform_loc(program, name)
        if loc >= 0:
            glUniform1f(loc, float(x))

    set_vec4("material.ambient", (1.0, 1.0, 1.0, 1.0))
    set_vec4("material.diffuse", (1.0, 1.0, 1.0, 1.0))
    set_vec4("material.specular", (0.6, 0.6, 0.6, 1.0))
    set_vec4("material.emission", (0.0, 0.0, 0.0, 1.0))
    set_f("material.shininess", 32.0)

    # light (точечный)
    set_vec4("light.ambient", (0.18, 0.18, 0.18, 1.0))
    set_vec4("light.diffuse", (0.95, 0.95, 0.95, 1.0))
    set_vec4("light.specular", (1.0, 1.0, 1.0, 1.0))
    set_vec3("light.attenuation", (1.0, 0.09, 0.032))

    # locs которые меняются каждый кадр
    loc_model = uniform_loc(program, "transform.model")
    loc_vp = uniform_loc(program, "transform.viewProjection")
    loc_normal = uniform_loc(program, "transform.normal")
    loc_viewpos = uniform_loc(program, "transform.viewPosition")
    loc_lightpos = uniform_loc(program, "light.position")

    # camera + light state
    cam_yaw = 0.8
    cam_pitch = -0.25
    cam_dist = 3.5

    light_pos = np.array([1.2, 1.2, 1.2], dtype=np.float32)

    last_t = glfw.get_time()

    while not glfw.window_should_close(window):
        t = glfw.get_time()
        dt = float(t - last_t)
        last_t = t

        glfw.poll_events()

        # выход
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        # camera: стрелки / PgUp PgDn
        cam_speed = 1.4 * dt
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            cam_yaw -= cam_speed
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            cam_yaw += cam_speed
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            cam_pitch += cam_speed
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            cam_pitch -= cam_speed
        cam_pitch = float(np.clip(cam_pitch, -1.2, 1.2))

        if glfw.get_key(window, glfw.KEY_PAGE_UP) == glfw.PRESS:
            cam_dist = max(0.5, cam_dist - 2.0 * dt)
        if glfw.get_key(window, glfw.KEY_PAGE_DOWN) == glfw.PRESS:
            cam_dist = min(50.0, cam_dist + 2.0 * dt)

        # light move: J/L (x), U/O (y), I/K (z)
        l_speed = 2.0 * dt
        if glfw.get_key(window, glfw.KEY_J) == glfw.PRESS:
            light_pos[0] -= l_speed
        if glfw.get_key(window, glfw.KEY_L) == glfw.PRESS:
            light_pos[0] += l_speed
        if glfw.get_key(window, glfw.KEY_U) == glfw.PRESS:
            light_pos[1] += l_speed
        if glfw.get_key(window, glfw.KEY_O) == glfw.PRESS:
            light_pos[1] -= l_speed
        if glfw.get_key(window, glfw.KEY_I) == glfw.PRESS:
            light_pos[2] -= l_speed
        if glfw.get_key(window, glfw.KEY_K) == glfw.PRESS:
            light_pos[2] += l_speed

        width, height = glfw.get_framebuffer_size(window)
        aspect = width / float(height if height else 1)

        # camera matrices
        eye = np.array([
            cam_dist * np.cos(cam_pitch) * np.sin(cam_yaw),
            cam_dist * np.sin(cam_pitch),
            cam_dist * np.cos(cam_pitch) * np.cos(cam_yaw),
        ], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        view = look_at(eye, target, up)
        proj = perspective(np.radians(45.0), aspect, 0.1, 200.0)
        vp = proj @ view

        # model matrix
        model = rotate_y(MODEL_ROT_SPEED * t) @ scale(MODEL_SCALE, MODEL_SCALE, MODEL_SCALE)

        # normal matrix (inverse transpose of model 3x3)
        m3 = model[:3, :3]
        normal_m = np.linalg.inv(m3).T.astype(np.float32)

        # draw
        glClearColor(0.03, 0.03, 0.04, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(program)

        # transforms
        if loc_model >= 0:
            glUniformMatrix4fv(loc_model, 1, GL_TRUE, model)
        if loc_vp >= 0:
            glUniformMatrix4fv(loc_vp, 1, GL_TRUE, vp)
        if loc_normal >= 0:
            glUniformMatrix3fv(loc_normal, 1, GL_TRUE, normal_m)
        if loc_viewpos >= 0:
            glUniform3f(loc_viewpos, float(eye[0]), float(eye[1]), float(eye[2]))

        # light position
        if loc_lightpos >= 0:
            glUniform4f(loc_lightpos, float(light_pos[0]), float(light_pos[1]), float(light_pos[2]), 1.0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, indices.size, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glBindTexture(GL_TEXTURE_2D, 0)

        glfw.swap_buffers(window)

    # cleanup
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteBuffers(1, [ebo])
    glDeleteTextures(1, [texture_id])
    glDeleteProgram(program)

    glfw.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
