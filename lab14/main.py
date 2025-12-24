# main.py
# Lab14 (часть 1): 1 объект + Point Light + Phong + Texture
# Dependencies: glfw, PyOpenGL, numpy, pillow

import sys
import math
import ctypes
import numpy as np

try:
    import glfw
    from OpenGL.GL import *
    from PIL import Image
except Exception as e:
    print("Не хватает библиотек. Установи:")
    print("  py -m pip install glfw PyOpenGL PyOpenGL_accelerate numpy pillow")
    print("Ошибка импорта:", e)
    sys.exit(1)


# ----------------------------- GLSL SHADERS -----------------------------

VERTEX_SHADER_SRC = """
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 vUV;

void main()
{
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;

    // корректное преобразование нормалей (если есть неравномерный масштаб)
    mat3 normalMat = transpose(inverse(mat3(uModel)));
    vWorldNormal = normalize(normalMat * aNormal);

    vUV = aUV;

    gl_Position = uProj * uView * worldPos;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core

in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 vUV;

out vec4 FragColor;

uniform sampler2D uTex;

uniform vec3 uViewPos;

// point light
uniform vec3 uLightPos;
uniform vec3 uLightAmbient;
uniform vec3 uLightDiffuse;
uniform vec3 uLightSpecular;
uniform vec3 uAtten; // (constant, linear, quadratic)

// material
uniform float uShininess;
uniform vec3  uSpecColor;

void main()
{
    vec3 N = normalize(vWorldNormal);
    vec3 L = uLightPos - vWorldPos;
    float dist = length(L);
    vec3 lightDir = normalize(L);

    vec3 V = normalize(uViewPos - vWorldPos);

    // затухание
    float attenuation = 1.0 / (uAtten.x + uAtten.y * dist + uAtten.z * dist * dist);

    // базовый цвет из текстуры
    vec3 texColor = texture(uTex, vUV).rgb;

    // ambient
    vec3 ambient = uLightAmbient * texColor;

    // diffuse
    float diff = max(dot(N, lightDir), 0.0);
    vec3 diffuse = uLightDiffuse * diff * texColor;

    // specular (Phong)
    vec3 R = reflect(-lightDir, N);
    float spec = pow(max(dot(V, R), 0.0), uShininess);
    vec3 specular = uLightSpecular * spec * uSpecColor;

    vec3 color = (ambient + diffuse + specular) * attenuation;
    FragColor = vec4(color, 1.0);
}
"""


# ----------------------------- UTILS: SHADERS -----------------------------

def compile_shader(src: str, shader_type) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)

    ok = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not ok:
        log = glGetShaderInfoLog(shader).decode("utf-8", errors="ignore")
        tname = "VERTEX" if shader_type == GL_VERTEX_SHADER else "FRAGMENT"
        raise RuntimeError(f"{tname} SHADER compile error:\n{log}")
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
        log = glGetProgramInfoLog(prog).decode("utf-8", errors="ignore")
        raise RuntimeError(f"PROGRAM link error:\n{log}")

    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


# ----------------------------- MATH: MATRICES -----------------------------

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def look_at(eye, target, up):
    f = normalize(target - eye)
    r = normalize(np.cross(f, up))
    u = np.cross(r, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = r
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(r, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def perspective(fovy_rad, aspect, z_near, z_far):
    f = 1.0 / math.tan(fovy_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (z_far + z_near) / (z_near - z_far)
    m[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    m[3, 2] = -1.0
    return m


def rotate_y(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([
        [ c, 0.0,  s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0,  c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)


def translate(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale(sx, sy, sz):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


# ----------------------------- TEXTURE -----------------------------

def create_checker_texture(size=256, cells=8):
    img = Image.new("RGB", (size, size))
    pix = img.load()
    cell = size // cells
    for y in range(size):
        for x in range(size):
            cx = x // cell
            cy = y // cell
            v = 255 if (cx + cy) % 2 == 0 else 40
            pix[x, y] = (v, v, v)
    return img


def load_texture_2d(path=None):
    if path:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = create_checker_texture()
    else:
        img = create_checker_texture()

    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    data = np.frombuffer(img.tobytes(), dtype=np.uint8)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB,
        img.width, img.height,
        0, GL_RGB, GL_UNSIGNED_BYTE,
        data
    )
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex


# ----------------------------- GEOMETRY: OBJ LOADER -----------------------------

def load_obj_simple(path):
    """
    Очень простой OBJ:
    Поддерживает v, vt, vn, f.
    Делает треугольники (если грань >3, делит веером).
    Возвращает плоский массив float32: [pos(3), normal(3), uv(2)] * vertex_count
    """
    positions = []
    uvs = []
    normals = []
    vertices = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == "v":
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "vt":
                uvs.append([float(parts[1]), float(parts[2])])
            elif tag == "vn":
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "f":
                face = parts[1:]
                idxs = []
                for vtx in face:
                    comps = vtx.split("/")
                    vi = int(comps[0]) - 1 if comps[0] else -1
                    ti = int(comps[1]) - 1 if len(comps) > 1 and comps[1] else -1
                    ni = int(comps[2]) - 1 if len(comps) > 2 and comps[2] else -1
                    idxs.append((vi, ti, ni))

                # triangulation: fan (0, i, i+1)
                for i in range(1, len(idxs) - 1):
                    tri = [idxs[0], idxs[i], idxs[i + 1]]
                    for (vi, ti, ni) in tri:
                        p = positions[vi] if vi >= 0 else [0.0, 0.0, 0.0]
                        t = uvs[ti] if ti >= 0 else [0.0, 0.0]
                        n = normals[ni] if ni >= 0 else [0.0, 1.0, 0.0]
                        vertices.extend([p[0], p[1], p[2], n[0], n[1], n[2], t[0], t[1]])

    arr = np.array(vertices, dtype=np.float32)
    return arr


def build_cube():
    """
    Куб из 12 треугольников (36 вершин).
    Формат: pos(3), normal(3), uv(2)
    """
    # позиции для каждой грани отдельно (чтобы нормали были "плоские")
    # uv: стандартная раскладка 0..1 на грань
    def face(p0, p1, p2, p3, n):
        # два треугольника: (0,1,2) и (0,2,3)
        uv0, uv1, uv2, uv3 = (0,0), (1,0), (1,1), (0,1)
        verts = []
        for (p, uv) in [(p0, uv0), (p1, uv1), (p2, uv2),
                        (p0, uv0), (p2, uv2), (p3, uv3)]:
            verts += [p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1]]
        return verts

    s = 0.5
    # 8 углов
    A = (-s, -s, -s)
    B = ( s, -s, -s)
    C = ( s,  s, -s)
    D = (-s,  s, -s)
    E = (-s, -s,  s)
    F = ( s, -s,  s)
    G = ( s,  s,  s)
    H = (-s,  s,  s)

    data = []
    # front (z+): E F G H
    data += face(E, F, G, H, (0, 0, 1))
    # back (z-): B A D C  (порядок чтобы нормаль смотрела наружу)
    data += face(B, A, D, C, (0, 0, -1))
    # left (x-): A E H D
    data += face(A, E, H, D, (-1, 0, 0))
    # right (x+): F B C G
    data += face(F, B, C, G, (1, 0, 0))
    # top (y+): D H G C
    data += face(D, H, G, C, (0, 1, 0))
    # bottom (y-): A B F E
    data += face(A, B, F, E, (0, -1, 0))

    return np.array(data, dtype=np.float32)


# ----------------------------- OPENGL BUFFERS -----------------------------

def create_vao_vbo(interleaved_vertices: np.ndarray):
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, interleaved_vertices.nbytes, interleaved_vertices, GL_STATIC_DRAW)

    stride = 8 * 4  # 8 floats * 4 bytes

    # pos (location=0)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

    # normal (location=1)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))

    # uv (location=2)
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    vertex_count = interleaved_vertices.size // 8
    return vao, vbo, vertex_count


# ----------------------------- MAIN -----------------------------

def main():
    if not glfw.init():
        print("GLFW init failed")
        return

    # OpenGL 3.3 core
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    width, height = 900, 600
    window = glfw.create_window(width, height, "Lab14: Phong + Point Light + Texture (1 object)", None, None)
    if not window:
        glfw.terminate()
        print("Window create failed")
        return

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # GL state
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # program
    program = create_program(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)

    # geometry: try model.obj, else cube
    tex_path = None
    try:
        vertices = load_obj_simple("model.obj")
        print("Loaded model.obj")
        tex_path = "texture.png"
    except Exception:
        vertices = build_cube()
        print("Using built-in cube (no model.obj found)")

    vao, vbo, vertex_count = create_vao_vbo(vertices)

    # texture
    tex = load_texture_2d(tex_path)

    # uniforms locations
    glUseProgram(program)
    loc_uModel = glGetUniformLocation(program, "uModel")
    loc_uView = glGetUniformLocation(program, "uView")
    loc_uProj = glGetUniformLocation(program, "uProj")

    loc_uTex = glGetUniformLocation(program, "uTex")
    glUniform1i(loc_uTex, 0)  # texture unit 0

    loc_uViewPos = glGetUniformLocation(program, "uViewPos")

    loc_uLightPos = glGetUniformLocation(program, "uLightPos")
    loc_uLightAmbient = glGetUniformLocation(program, "uLightAmbient")
    loc_uLightDiffuse = glGetUniformLocation(program, "uLightDiffuse")
    loc_uLightSpecular = glGetUniformLocation(program, "uLightSpecular")
    loc_uAtten = glGetUniformLocation(program, "uAtten")

    loc_uShininess = glGetUniformLocation(program, "uShininess")
    loc_uSpecColor = glGetUniformLocation(program, "uSpecColor")

    # camera
    cam_pos = np.array([0.0, 0.0, 2.2], dtype=np.float32)
    cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # light
    light_pos = np.array([1.2, 1.2, 1.2], dtype=np.float32)

    # material/light params
    light_ambient = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    light_diffuse = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    light_specular = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    atten = np.array([1.0, 0.09, 0.032], dtype=np.float32)  # typical point light attenuation

    shininess = 48.0
    spec_color = np.array([0.7, 0.7, 0.7], dtype=np.float32)

    angle = 0.0

    def on_resize(win, w, h):
        nonlocal width, height
        width, height = max(1, w), max(1, h)
        glViewport(0, 0, width, height)

    glfw.set_framebuffer_size_callback(window, on_resize)

    # simple controls:
    # arrows: move light in X/Z
    # W/S: move light in Y
    # A/D: rotate object
    def handle_input():
        nonlocal angle, light_pos
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            angle -= 0.02
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            angle += 0.02

        step = 0.03
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            light_pos[0] -= step
        if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            light_pos[0] += step
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            light_pos[2] -= step
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            light_pos[2] += step
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            light_pos[1] += step
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            light_pos[1] -= step

    while not glfw.window_should_close(window):
        glfw.poll_events()
        handle_input()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # matrices
        model = translate(0.0, 0.0, 0.0) @ rotate_y(angle) @ scale(1.0, 1.0, 1.0)
        view = look_at(cam_pos, cam_target, cam_up)
        proj = perspective(math.radians(60.0), width / float(height), 0.1, 100.0)

        glUseProgram(program)

        # set uniforms
        glUniformMatrix4fv(loc_uModel, 1, GL_TRUE, model)  # GL_TRUE => row-major to column-major convert
        glUniformMatrix4fv(loc_uView, 1, GL_TRUE, view)
        glUniformMatrix4fv(loc_uProj, 1, GL_TRUE, proj)

        glUniform3f(loc_uViewPos, cam_pos[0], cam_pos[1], cam_pos[2])

        glUniform3f(loc_uLightPos, light_pos[0], light_pos[1], light_pos[2])
        glUniform3f(loc_uLightAmbient, light_ambient[0], light_ambient[1], light_ambient[2])
        glUniform3f(loc_uLightDiffuse, light_diffuse[0], light_diffuse[1], light_diffuse[2])
        glUniform3f(loc_uLightSpecular, light_specular[0], light_specular[1], light_specular[2])
        glUniform3f(loc_uAtten, atten[0], atten[1], atten[2])

        glUniform1f(loc_uShininess, shininess)
        glUniform3f(loc_uSpecColor, spec_color[0], spec_color[1], spec_color[2])

        # draw
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)

        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, vertex_count)
        glBindVertexArray(0)

        glBindTexture(GL_TEXTURE_2D, 0)

        glfw.swap_buffers(window)

    # cleanup
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteTextures(1, [tex])
    glDeleteProgram(program)

    glfw.terminate()


if __name__ == "__main__":
    main()
