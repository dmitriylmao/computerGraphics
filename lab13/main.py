#py -m pip install pillow glfw PyOpenGL PyOpenGL_accelerate numpy

import sys
import ctypes

import glfw
from OpenGL.GL import *
import numpy as np
from PIL import Image

OBJ_PATH = "model.obj"
TEX_PATH = "texture.png"

VERTEX_SHADER_SRC = """
#version 330 core

layout(location = 0) in vec3 aPos;     // позиция вершины
layout(location = 1) in vec2 aTex;     // текстурные координаты

out vec2 vTex;

uniform mat4 uMVP;                     // матрица Model-View-Projection, отвечает за 3д

void main()
{
    gl_Position = uMVP * vec4(aPos, 1.0);
    vTex = aTex;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core

in vec2 vTex;
out vec4 FragColor;

uniform sampler2D uTexture;

void main()
{
    FragColor = texture(uTexture, vTex);
}
"""


def perspective(fovy_rad, aspect, z_near, z_far):
    """Перспективная матрица проекции."""
    f = 1.0 / np.tan(fovy_rad / 2.0)
    a = (z_far + z_near) / (z_near - z_far)
    b = (2.0 * z_far * z_near) / (z_near - z_far)

    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, a, b],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype=np.float32)


def translate(x, y, z):
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)


def scale(sx, sy, sz):
    return np.array([
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)


def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    status = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not status:
        info_log = glGetShaderInfoLog(shader).decode()
        raise RuntimeError("Ошибка компиляции шейдера:\n" + info_log)

    return shader


def load_obj(path):
    positions = []   # все вершины v x y z из файла
    texcoords = []   # все текстурные координаты vt u v;
    vertex_data = []  # итоговый массив, который уйдёт в VBO (позиция + текстура)
    indices = []   # индексы треугольников для EBO
    vert_map = {} # (vi, ti) -> индекс вершины в vertex_data

    with open(path, "r", encoding="utf-8") as f:
        for line in f: #Читаем .obj построчно
            if line.startswith("#") or len(line.strip()) == 0:
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "v" and len(parts) >= 4:     #v(список точек в 3D) → кладём в positions
                x, y, z = map(float, parts[1:4])
                positions.append((x, y, z))

            elif parts[0] == "vt" and len(parts) >= 3:      #vt(Текстурные координаты uv) → кладём в texcoords
                u, v = map(float, parts[1:3])
                texcoords.append((u, v))

            elif parts[0] == "f" and len(parts) >= 4:       #f → обрабатываем грань треугольника
                face_indices = []
            #f 1/1 2/2 3/3 — грань треугольника
            #Каждый «1/1» — это пара индексов позиция/текстурная_координата = vIndex/vtIndex.
            #первая вершина грани использует позицию v1 и текстуру vt1, вторая — v2 + vt2, и тд
                for token in parts[1:]:
                    # варианты: v/vt, v/vt/vn, v//vn
                    vals = token.split("/")
                    vi = int(vals[0])
                    ti = None
                    if len(vals) > 1 and vals[1] != "":
                        ti = int(vals[1])

                    # корректируем индексы потомут что могут быть отрицательные
                    if vi < 0:
                        vi = len(positions) + vi
                    else:
                        vi -= 1

                    if ti is not None:
                        if ti < 0:
                            ti = len(texcoords) + ti
                        else:
                            ti -= 1

                    #Склеиваем позицию и текстуру в одну вершину
                    key = (vi, ti)

                    if key not in vert_map:
                        px, py, pz = positions[vi]
                        if ti is not None and 0 <= ti < len(texcoords):
                            tu, tv = texcoords[ti]
                        else:
                            tu, tv = 0.0, 0.0

                        idx = len(vertex_data) // 5
                        vert_map[key] = idx
                        vertex_data.extend([px, py, pz, tu, tv])

                    face_indices.append(vert_map[key])

                # триангулируем многоугольник с помощью fan (любой н угольник превращаем в набор треугольников)
                for i in range(1, len(face_indices) - 1):
                    indices.extend([
                        face_indices[0],
                        face_indices[i],
                        face_indices[i + 1]
                    ])

    if not vertex_data or not indices:
        raise RuntimeError("Не удалось загрузить модель из OBJ: " + path)

    vertices_np = np.array(vertex_data, dtype=np.float32) #[x0, y0, z0, u0, v0 ....]
    indices_np = np.array(indices, dtype=np.uint32) #[0 1 2  ,  0 2 3  ....]
    return vertices_np, indices_np


def load_texture(path):
    img = Image.open(path).convert("RGBA")
    img_data = np.array(img)[::-1, :, :]  # переворачиваем по вертикали
    height, width, _ = img_data.shape

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        img_data
    )
    glGenerateMipmap(GL_TEXTURE_2D)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glBindTexture(GL_TEXTURE_2D, 0)
    return tex_id



def main():
    # GLFW 
    if not glfw.init():
        print("GLFW не инициализировался", file=sys.stderr)
        return

    window = glfw.create_window(800, 600, "Solar System", None, None)
    if not window:
        glfw.terminate()
        print("Не удалось создать окно", file=sys.stderr)
        return

    glfw.make_context_current(window)

    width, height = glfw.get_framebuffer_size(window)
    glViewport(0, 0, width, height)

    v_shader = compile_shader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER) #компиляция обоих шейдеров
    f_shader = compile_shader(FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER)

    shader_program = glCreateProgram()
    glAttachShader(shader_program, v_shader) #прикрепляем к пустой шейдерной программе оба шейдера
    glAttachShader(shader_program, f_shader)
    glLinkProgram(shader_program) #линкуем оба шейдера и проверяеем линковку

    status = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if not status:
        info_log = glGetProgramInfoLog(shader_program).decode()
        raise RuntimeError("Ошибка линковки программы:\n" + info_log)

    glDeleteShader(v_shader)
    glDeleteShader(f_shader)

    vertices, indices = load_obj(OBJ_PATH) # загрузка модели и текстуры
    index_count = indices.size

    texture_id = load_texture(TEX_PATH)

    VBO = glGenBuffers(1) # VBO + EBO
    EBO = glGenBuffers(1)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    stride = 5 * vertices.itemsize  # 3 pos + 2 tex

    glEnable(GL_DEPTH_TEST) #глубина (Z-буфер), чтобы дальние грани не рисовались поверх ближних

    projection = perspective(np.radians(30.0), width / float(height), 0.1, 100.0)
    view = translate(0.0, 0.0, -15.0)

    glUseProgram(shader_program) #активирум прогу и получаем uMVP из шейдра
    mvp_loc = glGetUniformLocation(shader_program, "uMVP")
    tex_loc = glGetUniformLocation(shader_program, "uTexture")
    glUniform1i(tex_loc, 0)  # текстура в слоте 0

    # экземпляры модели
    instances = [
        translate(0.0, 0.0, 0.0) @ scale(2.0, 2.0, 2.0),    # центр
        translate(-4.0, 0.0, 0.0) @ scale(1.2, 1.2, 1.2),
        translate(4.0, 0.0, 0.0) @ scale(1.2, 1.2, 1.2),
        translate(-2.0, 0.0, 0.0) @ scale(0.9, 0.9, 0.9),
        translate(2.0, 0.0, 0.0) @ scale(0.9, 0.9, 0.9),
        translate(0.0, 0.0, -3.0) @ scale(0.7, 0.7, 0.7),   
    ]

    # главный цикл
    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id) #активируем и привязывам текстуру

        glBindBuffer(GL_ARRAY_BUFFER, VBO) #привязываем vbo ebo
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)

        # атрибут 0  позиция
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT,
            GL_FALSE,
            stride,
            ctypes.c_void_p(0)
        )

        # атрибут 1  текстурные кодры u v 
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1,
            2,
            GL_FLOAT,
            GL_FALSE,
            stride,
            ctypes.c_void_p(3 * vertices.itemsize)
        )

        # рендерим несколько экземпляров
        for model in instances:
            mvp = projection @ view @ model
            glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, mvp)
            glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)

        glDisableVertexAttribArray(0) # Отключаем атрибуты и отвязываем буфер
        glDisableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

        glfw.swap_buffers(window)

    glDeleteBuffers(1, [VBO]) # очистка
    glDeleteBuffers(1, [EBO])
    glDeleteTextures([texture_id])
    glDeleteProgram(shader_program)
    glfw.terminate()


if __name__ == "__main__":
    main()
