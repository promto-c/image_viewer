from typing import Tuple, List
from OpenGL import GL
import numpy as np

from functools import wraps

class Texture2D:

    PIXEL_DATA_MAPPING = {
        np.dtype('uint8'): GL.GL_UNSIGNED_BYTE,
        np.dtype('float16'): GL.GL_FLOAT,
        np.dtype('float32'): GL.GL_FLOAT,
    }

    TEXTURE_FORMAT_MAPPING = {
        np.dtype('uint8'): GL.GL_RGB,
        np.dtype('float16'): GL.GL_RGB16F,
        np.dtype('float32'): GL.GL_RGB32F,
    }

    def __init__(self, image_data: np.ndarray):
        # Get the height and width of the image
        self.height, self.width, _channel = image_data.shape

        # Create an OpenGL texture ID and bind the texture
        self.id = GL.glGenTextures(1)

        self.set_image(image_data)

    def __enter__(self):
        self.bind()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def _use_texture(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.bind()
            result = func(self, *args, **kwargs)
            self.release()
            return result
        return wrapper

    @_use_texture
    def set_image(self, image_data: np.ndarray):
        # Set the texture minification/magnification filter
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        
        # Use glTexImage2D to set the image texture in OpenGL
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,                               # target
            0,                                              # level
            self.TEXTURE_FORMAT_MAPPING[image_data.dtype],  # internal format
            self.width, self.height,                        # width and height of the texture
            0,                                              # border (must be 0)
            GL.GL_RGB,                                      # format of the pixel data
            self.PIXEL_DATA_MAPPING[image_data.dtype],      # data type of the pixel data
            np.flipud(image_data)                           # flattened image data as a list
        )

    def bind(self):
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.id)

    def release(self):
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

class RectangleMesh:
    # Define vertices (Position: X, Y)
    VERTICES = np.array([
            [-1.0, -1.0],  # Bottom left
            [ 1.0, -1.0],  # Bottom right
            [ 1.0,  1.0],  # Top right
            [-1.0,  1.0]   # Top left
        ], dtype=np.float32)

    def __init__(self):
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.create_mesh()

    def create_mesh(self):
        # Bind VAO
        GL.glBindVertexArray(self.vao)
        # Bind VBO
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)

        # Send data to buffer
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.VERTICES.nbytes, self.VERTICES, GL.GL_STATIC_DRAW)

        # Define attributes
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)  # positions
        GL.glEnableVertexAttribArray(0)

        # Unbind VBO
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)  # unbind VBO
        # Unbind VAO
        GL.glBindVertexArray(0)

class Entity:
    def __init__(self):
        pass

    def render(self):
        raise NotImplementedError("Subclasses must implement the render method.")

class LayerEntity(Entity):
    def __init__(self):
        self.transformation_matrix = np.eye(4)
        self.children: List[Entity] = list()

    def render(self):
        # Apply the transformation matrix to the children
        for child in self.children:
            child.render()

    def add_child(self, child):
        self.children.append(child)

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)

    def clear_children(self):
        self.children.clear()

    def set_transformation_matrix(self, transformation_matrix):
        self.transformation_matrix = transformation_matrix

    def get_transformation_matrix(self):
        return self.transformation_matrix

class LineEntity(Entity):
    def __init__(self, line_start, line_end, line_width=2.0, line_color=(1.0, 0.0, 0.0)):
        self.line_start = line_start
        self.line_end = line_end
        self.line_width = line_width
        self.line_color = line_color

    def render(self):
        # Set the line width to the calculated pixel size
        GL.glLineWidth(self.line_width)

        # Draw the line
        GL.glColor3f(*self.line_color)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex2f(*self.line_start)
        GL.glVertex2f(*self.line_end)
        GL.glEnd()

        # Reset the color to white
        GL.glColor3f(1.0, 1.0, 1.0)

class CanvasEntity(Entity):
    def __init__(self):
        super().__init__()
        self.rectangle_mesh = RectangleMesh()
        self.vao = self.rectangle_mesh.vao
        self.texture = None

    def set_image(self, image_data: np.ndarray = None):
        if self.texture is None:
            self.texture = Texture2D(image_data) if image_data is not None else None
        else:
            self.texture.set_image(image_data)

    def render(self):
        if self.texture is None:
            return

        # Bind the texture to the current active texture unit
        with self.texture:
            GL.glBindVertexArray(self.vao)
            GL.glDrawArrays(GL.GL_QUADS, 0, 4)
            GL.glBindVertexArray(0)
