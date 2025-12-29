
import math
import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import glfw
from OpenGL.GL import *
from PIL import Image


HERE = Path(__file__).resolve().parent

ASSETS = {
    "airship": ("airship.obj", "airship.png"),
    "tree":    ("tree.obj", "tree.png"),
    "cloud":   ("cloud.obj", "cloud.png"),
    "balloon": ("balloon.obj", "balloon.png"),
    "ground":  (None, "ground.png"),
}


LIT_VS = r"""
#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aNorm;
layout (location=2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vPosW;
out vec3 vNormW;
out vec2 vUV;

void main()
{
    vec4 w = uModel * vec4(aPos, 1.0); 
    vPosW = w.xyz;

    mat3 nmat = transpose(inverse(mat3(uModel)));
    vNormW = normalize(nmat * aNorm);

    vUV = aUV;
    gl_Position = uProj * uView * w;
}
"""

LIT_FS = r"""
#version 330 core
in vec3 vPosW;
in vec3 vNormW;
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uTex;
uniform vec3 uViewPos;

// Directional (global)
uniform vec3  uDirLightDir;
uniform vec3  uDirLightColor;
uniform float uAmbientStrength;

// Phong
uniform float uShininess;
uniform vec3  uSpecColor;

// Spotlight (projector) [п.8]
uniform int   uUseSpot;
uniform vec3  uSpotPos;
uniform vec3  uSpotDir;
uniform vec3  uSpotColor;
uniform float uSpotIntensity;
uniform float uSpotCosInner;
uniform float uSpotCosOuter;

float sat(float x){ return clamp(x, 0.0, 1.0); }

void main()
{
    vec3 albedo = texture(uTex, vUV).rgb;

    vec3 N = normalize(vNormW);
    vec3 V = normalize(uViewPos - vPosW);

    //phong
    vec3 Ld = normalize(-uDirLightDir);
    float diff = max(dot(N, Ld), 0.0);
    vec3 R = reflect(-Ld, N);
    float spec = pow(max(dot(V, R), 0.0), uShininess);

    vec3 ambient  = uAmbientStrength * uDirLightColor * albedo;
    vec3 diffuse  = diff * uDirLightColor * albedo;
    vec3 specular = spec * uDirLightColor * uSpecColor;

    vec3 color = ambient + diffuse + specular;

    //spotlight
    if(uUseSpot == 1)
    {
        vec3 L = uSpotPos - vPosW;
        float dist = length(L);
        vec3 Ldir = normalize(L);

        float cosTheta = dot(normalize(-uSpotDir), Ldir);
        float cone = sat((cosTheta - uSpotCosOuter) / (uSpotCosInner - uSpotCosOuter));

        float diffS = max(dot(N, Ldir), 0.0);
        vec3 RS = reflect(-Ldir, N);
        float specS = pow(max(dot(V, RS), 0.0), uShininess);

        float att = 1.0 / (1.0 + 0.08*dist + 0.02*dist*dist);

        vec3 spot = (diffS * albedo + specS * uSpecColor) * uSpotColor * uSpotIntensity * cone * att;
        color += spot;
    }

    FragColor = vec4(color, 1.0);
}
"""

TARGET_VS = r"""
#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aNorm;
layout (location=2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec2 vUV;

void main()
{
    vUV = aUV;
    gl_Position = uProj * uView * (uModel * vec4(aPos, 1.0));
}
"""

TARGET_FS = r"""
#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform float uTime;

float ring(float r, float freq, float width)
{
    float x = fract(r * freq);
    return smoothstep(width, 0.0, x) + smoothstep(width, 0.0, 1.0 - x);
}

void main()
{
    // анимация
    vec2 wobble = vec2(cos(uTime), sin(uTime));
    vec2 center = vec2(0.5) + 0.25 * exp(-0.25 * uTime) * wobble;

    vec2 d = vUV - center;
    float r = length(d);

    // круг
    float alpha = smoothstep(0.52, 0.50, r);
    if(alpha <= 0.001) discard;

    float freq = 10.0 + 2.0 * sin(uTime * 1.5);
    float w = 0.03;

    float m = ring(r, freq, w);

    vec3 base = vec3(1.0);
    vec3 red  = vec3(0.90, 0.05, 0.05);
    vec3 col  = mix(base, red, clamp(m, 0.0, 1.0));

    FragColor = vec4(col, alpha);
}
"""


def compile_shader(src: str, stype: int) -> int:
    sh = glCreateShader(stype)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(sh).decode("utf-8", errors="ignore"))
    return sh


def make_program(vs: str, fs: str) -> int:
    v = compile_shader(vs, GL_VERTEX_SHADER)
    f = compile_shader(fs, GL_FRAGMENT_SHADER)
    p = glCreateProgram()
    glAttachShader(p, v)
    glAttachShader(p, f)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(p).decode("utf-8", errors="ignore"))
    glDeleteShader(v)
    glDeleteShader(f)
    return p


def load_texture(path: Path) -> int: #загрузка текстуры из пнг
    img = Image.open(path).convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
    data = np.frombuffer(img.tobytes(), dtype=np.uint8)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
    glGenerateMipmap(GL_TEXTURE_2D)

    glBindTexture(GL_TEXTURE_2D, 0)
    return tex



def _norm(v):
    n = float(np.linalg.norm(v))
    return v if n < 1e-8 else v / n


def perspective(fovy, aspect, zn, zf):
    f = 1.0 / math.tan(fovy * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zf + zn) / (zn - zf)
    m[2, 3] = (2.0 * zf * zn) / (zn - zf)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    f = _norm(target - eye)
    r = _norm(np.cross(f, up))
    u = np.cross(r, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = r
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -np.dot(r, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


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


def rotate_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  c,  -s,  0.0],
        [0.0,  s,   c,  0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def rotate_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [ c, 0.0,  s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0,  c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


@dataclass
class Mesh:
    vao: int
    vbo: int
    ebo: int
    index_count: int

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)


@dataclass
class Model:
    mesh: Mesh
    tex: int
    norm_mat: np.ndarray  


def make_mesh(vertices: np.ndarray, indices: np.ndarray) -> Mesh:
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    stride = 8 * 4  # pos3 + norm3 + uv2

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))

    glBindVertexArray(0)
    return Mesh(vao, vbo, ebo, int(indices.size))


def _compute_normals(pos: np.ndarray, idx: np.ndarray) -> np.ndarray: #если в обж не будет нормаоей
    n = np.zeros_like(pos, dtype=np.float32)
    tris = idx.reshape(-1, 3)
    for i0, i1, i2 in tris:
        p0, p1, p2 = pos[i0], pos[i1], pos[i2] #берем 3 точки
        fn = np.cross(p1 - p0, p2 - p0) #находим нормаль треугольника
        ln = float(np.linalg.norm(fn))
        if ln > 1e-8:
            fn /= ln
        n[i0] += fn
        n[i1] += fn
        n[i2] += fn
    for i in range(n.shape[0]):
        ln = float(np.linalg.norm(n[i]))
        if ln > 1e-8:
            n[i] /= ln
        else:
            n[i] = np.array([0, 1, 0], dtype=np.float32)
    return n


def load_obj(path: Path):
    v = [] #позиции вершин
    vt = [] #текстурные координаты
    vn = [] #нормали
    faces = [] # грани вида v/vt/vn

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v":
                v.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "vt":
                vt.append([float(parts[1]), float(parts[2])])
            elif tag == "vn":
                vn.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "f":
                faces.append(parts[1:])

    if not v or not faces:
        raise RuntimeError(f"OBJ пустой или без граней: {path.name}")

    v = np.array(v, dtype=np.float32)
    vt = np.array(vt, dtype=np.float32) if vt else None
    vn = np.array(vn, dtype=np.float32) if vn else None

    unique = {}
    out_pos, out_uv, out_n = [], [], []
    indices = []

    def parse_ref(tok: str):
        comps = tok.split("/")
        vi = int(comps[0]) - 1 if comps[0] else None
        ti = int(comps[1]) - 1 if len(comps) > 1 and comps[1] else None
        ni = int(comps[2]) - 1 if len(comps) > 2 and comps[2] else None
        return vi, ti, ni

    for face in faces:
        refs = [parse_ref(t) for t in face]
        for i in range(1, len(refs) - 1):
            tri = (refs[0], refs[i], refs[i + 1])
            for vi, ti, ni in tri:
                key = (vi, ti if vt is not None else None, ni if vn is not None else None)
                if key not in unique:
                    unique[key] = len(out_pos)
                    out_pos.append(v[vi].tolist())
                    if vt is not None and ti is not None and 0 <= ti < len(vt):
                        out_uv.append(vt[ti].tolist())
                    else:
                        out_uv.append([0.0, 0.0])
                    if vn is not None and ni is not None and 0 <= ni < len(vn):
                        out_n.append(vn[ni].tolist())
                    else:
                        out_n.append([0.0, 0.0, 0.0])
                indices.append(unique[key])

    pos = np.array(out_pos, dtype=np.float32)
    uv = np.array(out_uv, dtype=np.float32)
    nrm = np.array(out_n, dtype=np.float32)
    idx = np.array(indices, dtype=np.uint32)

    if vn is None or np.allclose(nrm, 0.0):
        nrm = _compute_normals(pos, idx) #если нормалей нет

    pmin = pos.min(axis=0)
    pmax = pos.max(axis=0)
    center = (pmin + pmax) * 0.5
    extent = float(np.max(pmax - pmin))
    if extent < 1e-6:
        extent = 1.0

    verts = np.hstack([pos, nrm, uv]).astype(np.float32).ravel()
    norm_mat = scale(1.0/extent, 1.0/extent, 1.0/extent) @ translate(-center[0], -center[1], -center[2])
    return verts, idx, norm_mat


def load_model(obj_name: str, tex_name: str) -> Model: #вспомогалка
    obj_path = HERE / obj_name
    tex_path = HERE / tex_name
    if not obj_path.exists():
        raise FileNotFoundError(f"Нет файла модели: {obj_name}")
    if not tex_path.exists():
        raise FileNotFoundError(f"Нет файла текстуры: {tex_name}")

    verts, idx, norm_mat = load_obj(obj_path)
    mesh = make_mesh(verts, idx)
    tex = load_texture(tex_path)
    return Model(mesh, tex, norm_mat)


def random_positions(rng, count, bounds, min_dist): #генератор точек на xz 
    xmin, xmax, zmin, zmax = bounds
    pts = []
    tries = 0
    while len(pts) < count and tries < 8000:
        tries += 1
        x = rng.uniform(xmin, xmax)
        z = rng.uniform(zmin, zmax)
        ok = True
        for px, pz in pts:
            if (x - px) ** 2 + (z - pz) ** 2 < min_dist ** 2:
                ok = False
                break
        if ok:
            pts.append((float(x), float(z)))
    return pts


def main():
    missing = []
    for _, (objf, texf) in ASSETS.items(): #проверяем файлики
        if objf is not None and not (HERE / objf).exists():
            missing.append(objf)
        if texf is not None and not (HERE / texf).exists():
            missing.append(texf)
    if missing:
        print("Не хватает файлов:")
        for m in missing:
            print("  -", m)
        return

    if not glfw.init():
        print("GLFW init failed")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    win = glfw.create_window(1200, 750, "Final Lab (OBJ): 7 + 8 + 11", None, None)
    if not win:
        glfw.terminate()
        print("Window create failed")
        return

    glfw.make_context_current(win)
    glfw.swap_interval(1)

    glEnable(GL_DEPTH_TEST) #режимы опенГЛ
    glClearColor(0.62, 0.67, 0.75, 1.0)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    prog_lit = make_program(LIT_VS, LIT_FS)
    prog_target = make_program(TARGET_VS, TARGET_FS)

    airship = load_model(*ASSETS["airship"])
    tree = load_model(*ASSETS["tree"])
    cloud = load_model(*ASSETS["cloud"])
    balloon = load_model(*ASSETS["balloon"])
    ground_tex = load_texture(HERE / ASSETS["ground"][1])

    size = 80.0
    uv_scale = 16.0
    hs = size * 0.5
    ground_vertices = np.array([ #создаем плоскость земли ВРучнуЮ!
        [-hs, 0, -hs,  0,1,0,  0,0],
        [ hs, 0, -hs,  0,1,0,  uv_scale,0],
        [ hs, 0,  hs,  0,1,0,  uv_scale,uv_scale],
        [-hs, 0,  hs,  0,1,0,  0,uv_scale],
    ], dtype=np.float32).ravel()
    ground_indices = np.array([0,1,2, 0,2,3], dtype=np.uint32)
    ground_mesh = make_mesh(ground_vertices, ground_indices)

    tq_v = np.array([ #мишени тоже в ручную
        [-0.5, 0, -0.5,  0,1,0,  0,0],
        [ 0.5, 0, -0.5,  0,1,0,  1,0],
        [ 0.5, 0,  0.5,  0,1,0,  1,1],
        [-0.5, 0,  0.5,  0,1,0,  0,1],
    ], dtype=np.float32).ravel()
    tq_i = np.array([0,1,2, 0,2,3], dtype=np.uint32)
    target_mesh = make_mesh(tq_v, tq_i)
    rng = np.random.RandomState(42)
    bounds = (-25, 25, -25, 25)


    target_pos = random_positions(rng, 5, bounds, min_dist=10.0) #случайные позиции объектов

    cloud_pos = random_positions(rng, 5, bounds, min_dist=10.0)
    balloon_pos = random_positions(rng, 6, bounds, min_dist=10.0)

    air_pos = np.array([0.0, 10.0, 15.0], dtype=np.float32)
    air_yaw = 0.0

    use_spot = True
    prev_f = False
#начальные параметры всего остального
    dir_light_dir = np.array([0.4, -1.0, 0.2], dtype=np.float32)
    dir_light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    ambient_strength = 0.22
    shininess = 48.0
    spec_color = np.array([0.6, 0.6, 0.6], dtype=np.float32)

    glUseProgram(prog_lit)
    uModel = glGetUniformLocation(prog_lit, "uModel")
    uView = glGetUniformLocation(prog_lit, "uView")
    uProj = glGetUniformLocation(prog_lit, "uProj")
    uTex = glGetUniformLocation(prog_lit, "uTex")
    uViewPos = glGetUniformLocation(prog_lit, "uViewPos")

    uDirLightDir = glGetUniformLocation(prog_lit, "uDirLightDir")
    uDirLightColor = glGetUniformLocation(prog_lit, "uDirLightColor")
    uAmbientStrength = glGetUniformLocation(prog_lit, "uAmbientStrength")

    uShininess = glGetUniformLocation(prog_lit, "uShininess")
    uSpecColor = glGetUniformLocation(prog_lit, "uSpecColor")

    uUseSpot = glGetUniformLocation(prog_lit, "uUseSpot")
    uSpotPos = glGetUniformLocation(prog_lit, "uSpotPos")
    uSpotDir = glGetUniformLocation(prog_lit, "uSpotDir")
    uSpotColor = glGetUniformLocation(prog_lit, "uSpotColor")
    uSpotIntensity = glGetUniformLocation(prog_lit, "uSpotIntensity")
    uSpotCosInner = glGetUniformLocation(prog_lit, "uSpotCosInner")
    uSpotCosOuter = glGetUniformLocation(prog_lit, "uSpotCosOuter")

    glUniform1i(uTex, 0)

    glUseProgram(prog_target)
    utModel = glGetUniformLocation(prog_target, "uModel")
    utView = glGetUniformLocation(prog_target, "uView")
    utProj = glGetUniformLocation(prog_target, "uProj")
    utTime = glGetUniformLocation(prog_target, "uTime")

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cam_offset = np.array([0.0, 6.0, 16.0], dtype=np.float32)

    def spot_params(): #прожектор привязан к дирижаблю и светит вниз
        spot_pos = air_pos + np.array([0.0, -1.2, 0.0], dtype=np.float32)
        spot_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        spot_color = np.array([1.0, 1.0, 0.95], dtype=np.float32)
        inner = math.cos(math.radians(12.0))
        outer = math.cos(math.radians(20.0))
        return spot_pos, spot_dir, spot_color, inner, outer

    def draw_model(m: Model, world_mat: np.ndarray):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, m.tex)
        model_mat = world_mat @ m.norm_mat
        glUniformMatrix4fv(uModel, 1, GL_TRUE, model_mat)
        m.mesh.draw()

    last_t = glfw.get_time()

    while not glfw.window_should_close(win):
        now = glfw.get_time()
        dt = float(now - last_t)
        last_t = now

        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS: #бинды кнопочек wasdqe f esc
            glfw.set_window_should_close(win, True)

        f_down = (glfw.get_key(win, glfw.KEY_F) == glfw.PRESS)
        if f_down and not prev_f:
            use_spot = not use_spot
        prev_f = f_down

        speed = 10.0
        v = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS: v[2] -= 1
        if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS: v[2] += 1
        if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS: v[0] -= 1
        if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS: v[0] += 1
        if glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS: v[1] -= 1
        if glfw.get_key(win, glfw.KEY_E) == glfw.PRESS: v[1] += 1
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        air_pos += v * speed * dt
        air_pos[0] = float(np.clip(air_pos[0], -35, 35))
        air_pos[2] = float(np.clip(air_pos[2], -35, 35))
        air_pos[1] = float(np.clip(air_pos[1], 4.0, 24.0))

        ww, hh = glfw.get_framebuffer_size(win)
        ww = max(1, ww)
        hh = max(1, hh)
        glViewport(0, 0, ww, hh)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        cam_pos = air_pos + cam_offset
        view = look_at(cam_pos, air_pos, up)
        proj = perspective(math.radians(60.0), ww / float(hh), 0.1, 300.0)

        glUseProgram(prog_lit)
        glUniformMatrix4fv(uView, 1, GL_TRUE, view)
        glUniformMatrix4fv(uProj, 1, GL_TRUE, proj)

        glUniform3f(uViewPos, cam_pos[0], cam_pos[1], cam_pos[2])

        glUniform3f(uDirLightDir, dir_light_dir[0], dir_light_dir[1], dir_light_dir[2])
        glUniform3f(uDirLightColor, dir_light_color[0], dir_light_color[1], dir_light_color[2])
        glUniform1f(uAmbientStrength, ambient_strength)

        glUniform1f(uShininess, shininess)
        glUniform3f(uSpecColor, spec_color[0], spec_color[1], spec_color[2])

        sp_pos, sp_dir, sp_col, sp_in, sp_out = spot_params()
        glUniform1i(uUseSpot, 1 if use_spot else 0)
        glUniform3f(uSpotPos, sp_pos[0], sp_pos[1], sp_pos[2])
        glUniform3f(uSpotDir, sp_dir[0], sp_dir[1], sp_dir[2])
        glUniform3f(uSpotColor, sp_col[0], sp_col[1], sp_col[2])
        glUniform1f(uSpotIntensity, 2.2)
        glUniform1f(uSpotCosInner, sp_in)
        glUniform1f(uSpotCosOuter, sp_out)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, ground_tex)
        glUniformMatrix4fv(uModel, 1, GL_TRUE, np.eye(4, dtype=np.float32))
        ground_mesh.draw()

        draw_model(tree, translate(0.0, 0.0, 0.0) @ scale(6.0, 6.0, 6.0))

        draw_model(airship, translate(air_pos[0], air_pos[1], air_pos[2]) @ rotate_y(air_yaw) @ scale(8.0, 8.0, 8.0))

        for x, z in cloud_pos:
            y = 18.0 + 2.0 * math.sin(now * 0.5 + x)
            draw_model(cloud, translate(x, y, z) @ scale(6.0, 6.0, 6.0))

        rot_fix = rotate_x(math.radians(270.0))
        for x, z in balloon_pos:
            y = 10.0 + 1.2 * math.sin(now * 0.8 + z)
            draw_model(balloon, translate(x, y, z) @ rot_fix @ scale(3.0, 3.0, 3.0))

        glUseProgram(prog_target)
        glUniformMatrix4fv(utView, 1, GL_TRUE, view)
        glUniformMatrix4fv(utProj, 1, GL_TRUE, proj)
        glUniform1f(utTime, float(now))

        for x, z in target_pos:
            m = translate(x, 0.03, z) @ scale(6.0, 1.0, 6.0)
            glUniformMatrix4fv(utModel, 1, GL_TRUE, m)
            target_mesh.draw()

        glfw.swap_buffers(win)

    glfw.terminate()


if __name__ == "__main__":
    main()
