from typing import Tuple, List
from OpenGL import GL
import numpy as np

from functools import wraps

import ctypes

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
            image_data                                      # image data
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

class ShapeEntity(Entity):
    def __init__(self, points: List[float], line_width=1.0, line_color=(1.0, 0.0, 0.0, 0.0), fill_color=None, is_bspline=True, degree=3, knot_vector=None):
        self.points = np.array(points)
        self.line_width = line_width
        self.line_color = line_color
        self.fill_color = fill_color
        self.is_bspline = is_bspline
        self.degree = degree
        self.show_control_points = True
        if is_bspline:
            if knot_vector:
                self.knot_vector = np.array(knot_vector)
            else:
                # Default to a uniform knot vector
                self.knot_vector = np.array([0]*degree + list(range(len(points) - degree + 1)) + [len(points) - degree]*(degree))

    def bspline_basis(self, i, k, t):
        if k == 0:
            return 1.0 if self.knot_vector[i] <= t < self.knot_vector[i + 1] else 0.0
        else:
            d1 = (t - self.knot_vector[i]) / (self.knot_vector[i + k] - self.knot_vector[i]) if self.knot_vector[i + k] - self.knot_vector[i] != 0 else 0
            d2 = (self.knot_vector[i + k + 1] - t) / (self.knot_vector[i + k + 1] - self.knot_vector[i + 1]) if self.knot_vector[i + k + 1] - self.knot_vector[i + 1] != 0 else 0
            return d1 * self.bspline_basis(i, k - 1, t) + d2 * self.bspline_basis(i + 1, k - 1, t)

    def get_bspline_point(self, t):
        point = np.zeros(2)
        for i, p in enumerate(self.points):
            point += self.bspline_basis(i, self.degree, t) * p
        return tuple(point)

    def toggle_control_points(self):
        """
        Toggle the visibility of control points.
        """
        self.show_control_points = not self.show_control_points

    def _create_vbo(self, data):
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, np.array(data, dtype=np.float32), GL.GL_STATIC_DRAW)
        return vbo

    def _draw_with_vbo(self, vbo, draw_mode, num_points):
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glDrawArrays(draw_mode, 0, num_points)
        GL.glDisableVertexAttribArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def render(self, shader_program):
        # Ensure blending and depth testing are disabled for 2D rendering
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_DEPTH_TEST)
        
        if self.is_bspline:
            num_points = 100
            curve_points = [self.get_bspline_point(t) for t in np.linspace(self.knot_vector[self.degree], self.knot_vector[-self.degree-1], num_points)]
        else:
            curve_points = self.points

        if self.fill_color:
            self._render_filled_shape(curve_points, shader_program)

        self._render_outline(curve_points, shader_program)
        
        if self.show_control_points:
            self._render_control_points(shader_program)

    def _cleanup_vbo(self, vbo):
        GL.glDeleteBuffers(1, [vbo])

    def _render_filled_shape(self, curve_points, shader_program):
        vbo = self._create_vbo(curve_points)
        # GL.glColor3f(*self.fill_color)
        shader_program.set_color(*self.fill_color)
        
        self._draw_with_vbo(vbo, GL.GL_POLYGON, len(curve_points))
        self._cleanup_vbo(vbo)

    def _render_outline(self, curve_points, shader_program):

        vbo = self._create_vbo(curve_points)
        GL.glLineWidth(self.line_width)
        # GL.glColor3f(*self.line_color)
        shader_program.set_color(*self.line_color)

        self._draw_with_vbo(vbo, GL.GL_LINE_LOOP, len(curve_points))
        self._cleanup_vbo(vbo)

    def _render_control_points(self, shader_program):
        vbo = self._create_vbo(self.points)
        GL.glPointSize(5.0)
        shader_program.set_color(0.1, 0.5, 0.8, 0.5)

        self._draw_with_vbo(vbo, GL.GL_POINTS, len(self.points))
        self._cleanup_vbo(vbo)

    def add_point(self, point):
        """
        Add a new control point to the shape.
        
        Args:
            point (tuple): A tuple (x, y) representing the control point to add.
        """
        self.points = np.vstack([self.points, np.array(point)])
        
        # If this shape is a B-spline, update the knot vector.
        if self.is_bspline:
            # Extend the knot vector by adding a new value at the end.
            # This is a simple approach, but depending on the specific use case, 
            # you might need more sophisticated methods to update the knot vector.
            max_knot_value = self.knot_vector[-1]
            new_knot_value = max_knot_value + 1
            self.knot_vector = np.append(self.knot_vector, new_knot_value)

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

    def render(self, shader_program):
        if self.texture is None:
            return
        
        shader_program.use_texture()

        # Bind the texture to the current active texture unit
        with self.texture:
            GL.glBindVertexArray(self.vao)
            GL.glDrawArrays(GL.GL_QUADS, 0, 4)
            GL.glBindVertexArray(0)
