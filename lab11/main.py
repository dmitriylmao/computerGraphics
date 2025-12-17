import sys
import ctypes

import glfw
from OpenGL.GL import *
import numpy as np



# aPos-позиция вершины
# aColor-цвет вершины
# Цвет пробрасываем через vColor
vertex_shader_src = """
#version 330 core

layout(location = 0) in vec2 aPos; 
layout(location = 1) in vec3 aColor;

out vec3 vColor;

void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}
"""


# выводит интерполированный цвет vColor
#FragColor — конечный цвет пикселя
fragment_shader_src = """
#version 330 core

in vec3 vColor;
out vec4 FragColor;

void main()
{
    FragColor = vec4(vColor, 1.0);
}
"""


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
    if not glfw.init():
        print("GLFW не инициализировался", file=sys.stderr)
        return

    window = glfw.create_window(800, 600, "Gradient Quad", None, None)
    if not window:
        glfw.terminate()
        print("Не удалось создать окно", file=sys.stderr)
        return

    glfw.make_context_current(window)



    vertex_shader = compile_shader(vertex_shader_src, GL_VERTEX_SHADER)#компиляция обоих шейдеров
    fragment_shader = compile_shader(fragment_shader_src, GL_FRAGMENT_SHADER)

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader) #прикрепляем к пустой шейдерной программе оба шейдера
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program) #линкуем оба шейдера и проверяеем линковку

    status = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if not status:
        info_log = glGetProgramInfoLog(shader_program).decode()
        raise RuntimeError("Ошибка линковки программы:\n" + info_log)

    glDeleteShader(vertex_shader) #удаляем шейдеры , т.к они уже в программе
    glDeleteShader(fragment_shader)

    # Описываем квадрат как 2 треугольника для каждой вершины (x, y, r, g, b)
    vertices = np.array([
        -0.5, -0.5, 1.0, 0.0, 0.0,  
         0.5, -0.5, 0.0, 1.0, 0.0, 
         0.5,  0.5, 0.0, 0.0, 1.0,  

        -0.5, -0.5, 1.0, 0.0, 0.0,  
         0.5,  0.5, 0.0, 0.0, 1.0, 
        -0.5,  0.5, 1.0, 1.0, 0.0  
    ], dtype=np.float32)



    VBO = glGenBuffers(1) #выделаяем буфер и отправляем данные а видеопамять
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    stride = 5 * vertices.itemsize  # Размер шага: 5 float на вершину 2 pos + 3 color



    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClearColor(0.0, 0.0, 0.0, 1.0) # Чёрный фон
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program) # Привязываем VBO и выбираем шейдерную программу 
        glBindBuffer(GL_ARRAY_BUFFER, VBO)

        # Атрибут 0 - позиция 
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,                      # location = 0
            2,                      # по 2 компоненты (x, y)
            GL_FLOAT,
            GL_FALSE,
            stride,                 # шаг между вершинами
            ctypes.c_void_p(0)      # смещение в буфере
        )

        # Атрибут 1 - цвет
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1,                      # location = 1
            3,                      # r, g, b
            GL_FLOAT,
            GL_FALSE,
            stride,
            ctypes.c_void_p(2 * vertices.itemsize)  # смещение после 2 float (x,y) , ь.е берем р г б
        )

        glDrawArrays(GL_TRIANGLES, 0, 6) # Рисуем 6 вершин , т.е два треуголльинка

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
