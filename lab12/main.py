import sys
import ctypes
import numpy as np
import glfw
from OpenGL.GL import *


vertex_shader_src = """
#version 330 core

layout(location = 0) in vec3 aPos;    // позиция вершины
layout(location = 1) in vec3 aColor;  // цвет вершины

out vec3 vColor;

uniform mat4 uMVP;                    // матрица Model-View-Projection, отвечает за 3д

void main()
{
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
}
"""

fragment_shader_src = """
#version 330 core

in vec3 vColor;
out vec4 FragColor;

void main()
{
    FragColor = vec4(vColor, 1.0);    // интерполированный цвет
}
"""


def perspective(fovy_rad, aspect, z_near, z_far): #aspect—отношение ширины окна к высоте, z_near, z_far — ближняя и дальняя плоскость по Z
    """матрица перспективной  проекции"""
    f = 1.0 / np.tan(fovy_rad / 2.0) #что то типо фокусног7о расстояния 
    a = (z_far + z_near) / (z_near - z_far) #коэффициенты, которые «зажимают» глубину 
    b = (2.0 * z_far * z_near) / (z_near - z_far) #между near и far в диапазон, удобный для OpenGL

    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, a, b],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype=np.float32)


def translate(x, y, z):
    """Матрица сдвига."""
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)


def rotate_x(angle_rad):
    """Поворот вокруг оси X."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c,   -s,  0.0],
        [0.0, s,    c,  0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)


def rotate_y(angle_rad):
    """Поворот вокруг оси Y."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [ c,  0.0, s,  0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s,  0.0, c,  0.0],
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


def main():
    #GLFW
    if not glfw.init():
        print("GLFW не инициализировался", file=sys.stderr)
        return

    window = glfw.create_window(1920, 1080, "Gradient Tetrahedron", None, None)
    if not window:
        glfw.terminate()
        print("Не удалось создать окно", file=sys.stderr)
        return

    glfw.make_context_current(window)
    width, height = glfw.get_framebuffer_size(window)
    glViewport(0, 0, width, height)


    v_shader = compile_shader(vertex_shader_src, GL_VERTEX_SHADER)#компиляция обоих шейдеров
    f_shader = compile_shader(fragment_shader_src, GL_FRAGMENT_SHADER)

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


    #в тетрайдере 4 треугольника -> 12 вершин , каждая вршина x y z ,r g b
    vertices = np.array([
        0.0,  1.0,  0.0,   1.0, 0.0, 0.0,   
       -1.0, -1.0,  1.0,   0.0, 1.0, 0.0,    
        1.0, -1.0,  1.0,   0.0, 0.0, 1.0,  

        0.0,  1.0,  0.0,   1.0, 0.0, 0.0,
        1.0, -1.0,  1.0,   0.0, 0.0, 1.0,
        0.0, -1.0, -1.0,   1.0, 1.0, 0.0,   

        0.0,  1.0,  0.0,   1.0, 0.0, 0.0,
        0.0, -1.0, -1.0,   1.0, 1.0, 0.0,
       -1.0, -1.0,  1.0,   0.0, 1.0, 0.0,

       -1.0, -1.0,  1.0,   0.0, 1.0, 0.0,
        0.0, -1.0, -1.0,   1.0, 1.0, 0.0,
        1.0, -1.0,  1.0,   0.0, 0.0, 1.0,
    ], dtype=np.float32)


    VBO = glGenBuffers(1) #выделаяем буфер и отправляем данные а видеопамять
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    stride = 6 * vertices.itemsize  # 3 pos + 3 color

    glEnable(GL_DEPTH_TEST)  #глубина (Z-буфер), чтобы дальние грани не рисовались поверх ближних

    projection = perspective(np.radians(45.0), width / float(height), 0.1, 100.0) # Проекция
    base_rotation = rotate_y(np.radians(30.0)) @ rotate_x(np.radians(-30.0)) #начальный поворот на 45град

    offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # вектор смещения объекта по осям w a s d q e

    
    glUseProgram(shader_program)#активирум прогу и получаем uMVP из шейдра
    mvp_loc = glGetUniformLocation(shader_program, "uMVP")

    move_speed = 0.005

    while not glfw.window_should_close(window):
        glfw.poll_events()

        #A/D - X,    Q/E - Y,    W/S - Z
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            offset[0] -= move_speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            offset[0] += move_speed
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            offset[1] += move_speed
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            offset[1] -= move_speed
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            offset[2] += move_speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            offset[2] -= move_speed

        glClearColor(0.0, 0.0, 0.0, 1.0) # Чёрный фон
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader_program)



        model = translate(offset[0], offset[1], offset[2]) @ base_rotation #двигаем по осям с фикс наклоном тетраэдэра
        view = translate(0.0, 0.0, -5.0) # камера слегка отодвинута назад по Z
        mvp = projection @ view @ model #перспективная проекция

        glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, mvp) # передаём матрицу в шейдер и транспонируем ее

        glBindBuffer(GL_ARRAY_BUFFER, VBO)

        # атрибут 1 - позиуия
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT,
            GL_FALSE,
            stride,
            ctypes.c_void_p(0)
        )

        # цвет
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1,
            3,
            GL_FLOAT,
            GL_FALSE,
            stride,
            ctypes.c_void_p(3 * vertices.itemsize)
        )

        glDrawArrays(GL_TRIANGLES, 0, 12) # 12 вершин = 4 треугольника тетраэдра
         
        glDisableVertexAttribArray(0) # Отключаем атрибуты и отвязываем буфер
        glDisableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glfw.swap_buffers(window)

    #освобождение ресов
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(shader_program)
    glfw.terminate()


if __name__ == "__main__":
    main()
