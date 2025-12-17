#python -m pip install glfw PyOpenGL PyOpenGL_accelerate numpy

#GLSL — OpenGL Shading Language, язык для шейдеров
#VBO = Vertex Buffer Object — буфер вершин

import glfw
from OpenGL.GL import *
import numpy as np

vertex_shader_src = """
#version 330 core

layout(location = 0) in vec3 aPos;

void main() {
    gl_Position = vec4(aPos.x,aPos.y-0.5, aPos.z, 1);
}
"""

fragment_shader_src = """
#version 330 core

out vec4 FragColor;

void main() {
    FragColor = vec4(0.0, 1.0, 0.0, 1.0); 
}
"""



def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type) #создаём шейдер (вершинный или фрагментный)
    glShaderSource(shader, source)#передаём текст в шейдер
    glCompileShader(shader)#компилируем

    success = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not success: #проверка ошибок
        info_log = glGetShaderInfoLog(shader).decode()
        raise Exception(f"Ошибка компиляции шейдера:\n{info_log}")

    return shader


def main():
    if not glfw.init(): #запуск GLFW
        raise Exception("GLFW не инициализировался!")


    window = glfw.create_window(800, 600, "Green Triangle", None, None) #создание окна
    if not window:
        glfw.terminate()
        raise Exception("Невозможно создать окно!")

    glfw.make_context_current(window)


    vertex_shader = compile_shader(vertex_shader_src, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_src, GL_FRAGMENT_SHADER)


    shader_program = glCreateProgram() 
    glAttachShader(shader_program, vertex_shader) #прикрепляем к пустой шейдерной программе оба шейдера
    glAttachShader(shader_program, fragment_shader) 
    glLinkProgram(shader_program) #линкуем оба шейдера и проверяеем линковку
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if not success:
        info_log = glGetProgramInfoLog(shader_program)
        raise Exception(f"Ошибка линковки программы:\n{info_log}")


    glDeleteShader(vertex_shader) #удаляем шейдеры , т.к они уже в программе
    glDeleteShader(fragment_shader)


    triangle = np.array([
        -0.5, -0.5, 0.0,
         0.5, -0.5, 0.0,
         0.0,  0.5, 0.0
    ], dtype=np.float32)


    VBO = glGenBuffers(1)#выделаяем буфер и отправляем данные а видеопамять
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, triangle.nbytes, triangle, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Главный цикл
    while not glfw.window_should_close(window):
        glfw.poll_events() #обработка инпута и событий окна

        glClearColor(0, 0, 0, 1) #заливаем фон в черный
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program)#вкючаем шейдерную программу

        glBindBuffer(GL_ARRAY_BUFFER, VBO)#привязываем буфер

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)#apos чиатет из буфера(вбо) подряд каждые 3 float, т.е одну вершину

        glDrawArrays(GL_TRIANGLES, 0, 3) #отрисовка

        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glfw.swap_buffers(window)

    #освобождение ресов
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(shader_program)
    glfw.terminate()


if __name__ == "__main__":
    main()
