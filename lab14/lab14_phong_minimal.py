import sys
import ctypes
import math
from pathlib import Path

import numpy as np
import glfw
from OpenGL.GL import *
from PIL import Image

OBJ_PATH = "model.obj"       
TEX_PATH = "texture.png"      

WIN_W, WIN_H = 1280, 720

FOV_Y_DEG = 55.0
Z_NEAR, Z_FAR = 0.05, 500.0

AMBIENT_STRENGTH = 0.12
LIGHT_INTENSITY = 1.5
SHININESS = 64.0


VERT_SRC = r"""
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTex;

out vec3 vFragPos;
out vec3 vNormal;
out vec2 vTex;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

void main() {
    vec4 world = uModel * vec4(aPos, 1.0); //перевод из model space → world space
    vFragPos = world.xyz;

    // correct normal transform (handles scaling / rotation)
    vNormal = mat3(transpose(inverse(uModel))) * aNormal; //преобразование нормалей
    vTex = aTex;
    gl_Position = uProj * uView * world;
}
"""

FRAG_SRC = r"""
#version 330 core

in vec3 vFragPos;
in vec3 vNormal;
in vec2 vTex;

out vec4 FragColor;

uniform sampler2D uTex;

uniform vec3  uLightPos;
uniform vec3  uLightColor; //свет
uniform float uLightIntensity;

uniform vec3  uViewPos; //камера

uniform float uAmbientStrength;
uniform float uShininess;

void main() {
    vec3 albedo = texture(uTex, vTex).rgb;

    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightPos - vFragPos);
    vec3 V = normalize(uViewPos - vFragPos); //направление на камеру
    vec3 R = reflect(-L, N); //отражённый луч

    float diff = max(dot(N, L), 0.0); //диффузная

    float spec = 0.0;
    if (diff > 0.0) { //Блик
        spec = pow(max(dot(R, V), 0.0), uShininess);
    }

    //Сборка цвета
    vec3 ambient  = uAmbientStrength * albedo;
    vec3 diffuse  = diff * albedo * uLightColor;
    vec3 specular = spec * uLightColor;

    vec3 color = ambient + (diffuse + specular) * uLightIntensity;

    FragColor = vec4(color, 1.0);
}
"""

def compile_shader(src: str, shader_type: int) -> int:
    sid = glCreateShader(shader_type) #создание объекта шейдера, загрузка в него текста и комплиляция
    glShaderSource(sid, src)
    glCompileShader(sid)
    ok = glGetShaderiv(sid, GL_COMPILE_STATUS)
    if not ok:
        log = glGetShaderInfoLog(sid).decode("utf-8", errors="ignore")
        glDeleteShader(sid)
        raise RuntimeError(log)
    return sid


def create_program(vs_src: str, fs_src: str) -> int:
    vs = compile_shader(vs_src, GL_VERTEX_SHADER) #компиляция обоих шейдеров
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    pid = glCreateProgram()
    glAttachShader(pid, vs) #прикрепляем к пустой шейдерной программе оба шейдера
    glAttachShader(pid, fs)
    glLinkProgram(pid) #линкуем и проверяеем линковку
    ok = glGetProgramiv(pid, GL_LINK_STATUS)
    glDeleteShader(vs) #удаляем шейдеры после линк
    glDeleteShader(fs)  
    if not ok:
        log = glGetProgramInfoLog(pid).decode("utf-8", errors="ignore")
        glDeleteProgram(pid)
        raise RuntimeError(log)
    return pid


def load_texture(path: str) -> int: #агрузка png в OpenGL texture2D
    img = Image.open(path).convert("RGBA")
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    data = np.frombuffer(img.tobytes(), dtype=np.uint8)
    w, h = img.size

    tid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tid)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

    glGenerateMipmap(GL_TEXTURE_2D)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glBindTexture(GL_TEXTURE_2D, 0)
    return tid


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n

#проекция задаёт перспективу, FOV, aspect, near/far
def perspective(fovy_deg: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    fovy = math.radians(fovy_deg)
    f = 1.0 / math.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (z_far + z_near) / (z_near - z_far)
    m[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m

#корректировка индексов 
def _fix_index(i: int, n: int) -> int:
    return (n + i) if i < 0 else (i - 1)


def load_obj_strict(path: str):
    positions = [] # все вершины v x y z из файла
    texcoords = [] # все текстурные координаты vt u v;
    normals = []

    unique = {}   
    vertices = []
    indices = []

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OBJ not found: {path}")

    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.strip().split()
        if not parts:
            continue

        tag = parts[0]
        if tag == "v":
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif tag == "vt":
            texcoords.append([float(parts[1]), float(parts[2])])
        elif tag == "vn":
            normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif tag == "f":
            if len(parts) < 4: #Триангуляция
                continue

            face = parts[1:]
            for i in range(1, len(face) - 1):
                tri = [face[0], face[i], face[i + 1]]
                for vtx in tri:
                    comps = vtx.split("/") #нормали учитываются и парсятся из файла
                    if len(comps) < 3 or comps[1] == "" or comps[2] == "":
                        raise ValueError("OBJ must contain vt and vn for every face vertex (v/vt/vn).")

                    vi = _fix_index(int(comps[0]), len(positions))
                    ti = _fix_index(int(comps[1]), len(texcoords))
                    ni = _fix_index(int(comps[2]), len(normals))

                    key = (vi, ti, ni) #Склейка индексов OBJ в единые вершины OpenGL
                    if key not in unique:
                        pos = positions[vi]
                        nrm = normals[ni]
                        uv = texcoords[ti]
                        unique[key] = len(vertices)
                        vertices.append([pos[0], pos[1], pos[2], nrm[0], nrm[1], nrm[2], uv[0], uv[1]])
                    indices.append(unique[key])

    if not positions:
        raise ValueError("OBJ has no vertices (v).")
    if not texcoords:
        raise ValueError("OBJ has no texture coords (vt).")
    if not normals:
        raise ValueError("OBJ has no normals (vn).")
    if not indices:
        raise ValueError("OBJ has no faces (f).")

    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)

    pos = np.array(positions, dtype=np.float32)
    vmin = pos.min(axis=0)
    vmax = pos.max(axis=0)
    center = (vmin + vmax) * 0.5
    radius = float(np.linalg.norm(vmax - vmin)) * 0.5
    radius = max(radius, 1e-3)

    return vertices, indices, center.astype(np.float32), radius


def main():
    if not Path(OBJ_PATH).exists(): #Проверка файлов
        print(f"[ERROR] OBJ file not found: {OBJ_PATH}")
        print("Put your OBJ next to this script or change OBJ_PATH.")
        sys.exit(1)
    if not Path(TEX_PATH).exists():
        print(f"[ERROR] Texture file not found: {TEX_PATH}")
        print("Put your texture next to this script or change TEX_PATH.")
        sys.exit(1)

    if not glfw.init():
        print("[ERROR] glfw.init() failed")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(WIN_W, WIN_H, "Lab14: Phong + Point Light", None, None)
    if not window:
        glfw.terminate()
        print("[ERROR] glfw.create_window() failed")
        sys.exit(1)

    glfw.make_context_current(window)

    def on_resize(_win, w, h): #ресайз окна
        glViewport(0, 0, max(1, w), max(1, h))
    glfw.set_framebuffer_size_callback(window, on_resize)

    def on_key(_win, key, _sc, action, _mods): #закрытие окна по ESC
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
    glfw.set_key_callback(window, on_key)

    glEnable(GL_DEPTH_TEST) #для четкости модели , чтоб ближние полигоны перекрывали дальние

    vertices, indices, center, radius = load_obj_strict(OBJ_PATH) #загрузка модели
    tex_id = load_texture(TEX_PATH)

    vao = glGenVertexArrays(1) #хранит состояние атрибутов вершин
    vbo = glGenBuffers(1)   #хранит вершины
    ebo = glGenBuffers(1) #инексы вершин

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    stride = 8 * 4  
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)) #атрибут 0 aPos — 3 float
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4)) #атрибут 1 aNormal — 3 float
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4)) #атрибут 2 aTex — 2 float

    glBindVertexArray(0)

    program = create_program(VERT_SRC, FRAG_SRC) #Создание шейдерной программы



    #ставим камеру , свет и модель
    cam_pos = center + np.array([0.0, radius * 0.25, radius * 2.8], dtype=np.float32)
    target = center
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    light_pos = center + np.array([radius * 1.2, radius * 1.4, radius * 1.0], dtype=np.float32)
    light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    model = np.eye(4, dtype=np.float32)
    model[0, 3] = -center[0]
    model[1, 3] = -center[1]
    model[2, 3] = -center[2]

    glUseProgram(program)
    loc_model = glGetUniformLocation(program, "uModel") #получение локаций юниформов внутри шейдера
    loc_view  = glGetUniformLocation(program, "uView")
    loc_proj  = glGetUniformLocation(program, "uProj")
    loc_tex   = glGetUniformLocation(program, "uTex")

    loc_lpos  = glGetUniformLocation(program, "uLightPos")
    loc_lcol  = glGetUniformLocation(program, "uLightColor")
    loc_lint  = glGetUniformLocation(program, "uLightIntensity")

    loc_vpos  = glGetUniformLocation(program, "uViewPos")
    loc_amb   = glGetUniformLocation(program, "uAmbientStrength")
    loc_shin  = glGetUniformLocation(program, "uShininess")

    glUniform1i(loc_tex, 0)

    glUniform3fv(loc_lcol, 1, light_color)
    glUniform1f(loc_lint, float(LIGHT_INTENSITY))
    glUniform1f(loc_amb, float(AMBIENT_STRENGTH))
    glUniform1f(loc_shin, float(SHININESS))

    while not glfw.window_should_close(window):
        glfw.poll_events()

        w, h = glfw.get_framebuffer_size(window)
        aspect = float(w) / float(h if h > 0 else 1)

        view = look_at(cam_pos, target, up)
        proj = perspective(FOV_Y_DEG, aspect, Z_NEAR, Z_FAR)

        glClearColor(0.08, 0.08, 0.09, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(program)

        glUniformMatrix4fv(loc_model, 1, GL_TRUE, model) 
        glUniformMatrix4fv(loc_view,  1, GL_TRUE, view)
        glUniformMatrix4fv(loc_proj,  1, GL_TRUE, proj)

        glUniform3fv(loc_lpos, 1, light_pos)
        glUniform3fv(loc_vpos, 1, cam_pos)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, indices.size, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glfw.swap_buffers(window)


    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteBuffers(1, [ebo])
    glDeleteTextures(1, [tex_id])
    glDeleteProgram(program)

    glfw.terminate()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)
