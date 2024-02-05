# Standard Library Imports
# ------------------------
import copy
from functools import wraps
from numbers import Number
import numpy as np
from typing import Any, Dict, Tuple, List, Union, Type, Optional, TYPE_CHECKING

# Third Party Imports
# -------------------
from OpenGL import GL

# Local Imports
# -------------
if TYPE_CHECKING:
    from viewer import ImageViewerGLWidget

# VBO
# ---
def create_vbo(data):
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, np.array(data, dtype=np.float32), GL.GL_STATIC_DRAW)
    return vbo

def draw_with_vbo(vbo, draw_mode, num_points):
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glEnableVertexAttribArray(0)
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
    GL.glDrawArrays(draw_mode, 0, num_points)
    GL.glDisableVertexAttribArray(0)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

def draw_points(points: List[Tuple[float, float]], point_size: float = 5.0) -> np.uintc:
    vbo = create_vbo(points)
    GL.glPointSize(point_size)
    draw_with_vbo(vbo, GL.GL_POINTS, len(points))

    return vbo

def cleanup_vbo(vbo):
    GL.glDeleteBuffers(1, [vbo])

class Texture2D:

    PIXEL_DATA_MAPPING = {
        np.dtype('uint8'): GL.GL_UNSIGNED_BYTE,
        np.dtype('uint16'): GL.GL_UNSIGNED_BYTE,
        np.dtype('float16'): GL.GL_FLOAT,
        np.dtype('float32'): GL.GL_FLOAT,
    }

    TEXTURE_FORMAT_MAPPING = {
        np.dtype('uint8'): GL.GL_RGB,
        np.dtype('uint16'): GL.GL_RGB,
        np.dtype('float16'): GL.GL_RGB16F,
        np.dtype('float32'): GL.GL_RGB32F,
    }

    def __init__(self, width, height):
        # Get the height and width of the image
        self.width = width
        self.height = height

        # Create an OpenGL texture ID and bind the texture
        self.id = GL.glGenTextures(1)

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
    
    def set_width(self, width):
        self.width = width

    def set_height(self ,height):
        self.height = height
    
    def set_texture_size(self, width, height):
        self.set_width(width)
        self.set_height(height)

    def get_texture_size(self):
        return self.width, self.height

    @_use_texture
    def set_image(self, image_data: Optional[np.ndarray] = None):
        if image_data is None:
            return

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

# Property
# --------
class PropertyMeta(type):
    def __getitem__(self, item):
        return self

class Property(metaclass=PropertyMeta):
    def __init__(self, type: Type, default_value: Optional[Any] = None, value: Any = None, frame: float = None):
        self._type = type
        self._values = {None: default_value}

        self._default_value = default_value

        if value is not None:
            self.set_value(value, frame)

    def __contains__(self, frame: float) -> bool:
        """Check if a value is set for the given frame.
        """
        return frame in self._values

    def __getitem__(self, frame: float) -> Any:
        """Get the value for a given frame.
        """
        return self.get_value(frame)

    def __setitem__(self, frame: float, value: Any):
        """Set the value for a given frame.
        """
        self.set_value(value, frame)

    @property
    def type(self) -> Type:
        return self._type

    def get_value(self, frame: Optional[float] = None) -> Any:
        # value = self._values.get(frame)
        if frame not in self._values:
            self._values[frame] = copy.deepcopy(self._default_value)

        return self._values[frame]

    def set_value(self, value, frame: Optional[float] = None):
        self._values[frame] = value

    def is_key(self, frame: float) -> bool:
        """Check if a specific frame is a keyframe.
        """
        return frame in self._values

    def is_animated(self):
        return len(self.keys()) > 1

    def num_keys(self):
        return len(self.keys())

    def keys(self) -> List[float]:
        """Get a list of all frames that have key values or changes.
        """
        keys = list(self._values.keys())
        keys.remove(None)

        return keys

    def delete_value(self, frame: float) -> bool:
        """Delete the value for a given frame.
        """
        if frame in self._values:
            del self._values[frame]
            return True
        return False
        
    def range(self) -> Tuple[Optional[float], Optional[float]]:
        """Get the range of keyframes as (first, last). If no keyframes, return (None, None).
        """
        if not self.keys():
            return (None, None)
        return (min(self.keys()), max(self.keys()))

    def reset(self, value: Optional[Any] = None):
        """Reset the property to the given value and the is_animated flag to its initial state.
        If no value is provided, use the default value.
        """
        self._values = {None: value or self._values.get(None)}

class Properties:
    def __init__(self, **kwargs):
        self._props = {}
        self.add(**kwargs)

    def add(self, **kwargs):
        for key, value in kwargs.items():
            if not isinstance(value, Property):
                raise ValueError(f"Expected a Property instance for key '{key}', got {type(value)}")
            self._props[key] = value

    def __getattr__(self, key):
        return self._props.get(key)

    def __setattr__(self, key, value):
        if key in ['_props']:
            super().__setattr__(key, value)
        else:
            self._props[key] = value

    def get_property_names(self) -> List[str]:
        """Returns the names of all properties."""
        return list(self._props.keys())

    def has_property(self, prop_name: str) -> bool:
        """Checks if a particular property exists."""
        return prop_name in self._props

    def remove_value (self, prop_name: str):
        """Removes a property."""
        if prop_name in self._props:
            del self._props[prop_name]

    def get_property(self, prop_name: str) -> Optional[Property]:
        """Fetches a Property object by name."""
        return self._props.get(prop_name)

    def set_property(self, prop_name: str, prop: Property):
        """Sets a Property object by name."""
        self._props[prop_name] = prop

    def property_types(self) -> Dict[str, Type]:
        """Returns a dictionary mapping property names to their types."""
        return {name: prop.type for name, prop in self._props.items()}

# Entity
# ------
class Entity:

    props = Properties()

    def __init__(self):
        pass

    def render(self, frame: Union[float, None], viewer: 'ImageViewerGLWidget'):
        raise NotImplementedError("Subclasses must implement the render method.")

class LayerEntity(Entity):
    def __init__(self):
        self.transformation_matrix = np.eye(4)
        self.children: List[Entity] = list()

    def render(self, frame: Union[float, None], viewer: 'ImageViewerGLWidget'):
        # Apply the transformation matrix to the children
        for child in self.children:
            child.render(frame, viewer.shader_program)

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


class TrackPointEntity(Entity):

    def __init__(self, position: Tuple[float, float], frame: float, color=(0.0, 1.0, 0.0, 1.0), point_size=5.0):
        """Initializes the tracker entity.
        """
        self.props = Properties(
            position = Property(
                type=Tuple[float, float], 
                default_value=(0.0, 0.0), 
            ),
            color = Property(
                type=Tuple[float, float, float, float], 
                default_value=(0.0, 0.0, 0.0, 1.0),
            ),
            error = Property(
                type=float,
            ),
        )

        self.props.position.set_value(position, frame)
        self.props.color.set_value(color)

        # TODO: This should be global appearance's property instead of entity's property
        self.point_size = point_size

    def render(self, frame: Union[float, None], viewer: 'ImageViewerGLWidget'):
        
        position = self.props.position.get_value(frame)

        if position is None:
            return

        color = self.props.color.get_value()
        gl_position = viewer.canvas_to_gl_coords(position)

        viewer.shader_program.set_color(*color)
        vbo = draw_points([gl_position], point_size=self.point_size)

        cleanup_vbo(vbo)

    def set_position(self, position: Tuple[float, float], frame):
        self.props.position.set_value(position, frame)

    def get_position(self, frame):
        return self.props.position.get_value(frame)

class TrackPointsEntity(Entity):

    def __init__(self, point_size=5.0, track_color=(0.0, 1.0, 0.0, 0.5)):
        """
        Initializes the tracker entity.

        Args:
        - track_points (List[Tuple[float, float]]): A list of (x, y) points representing the tracked object's location over time.
        - point_size (float): Size of the tracked points when rendered.
        - track_color (Tuple[float, float, float, float]): Color of the track.
        """
        self.props = Properties(
            color = Property(
                type=Tuple[float, float, float, float], 
                default_value=(0.0, 0.0, 0.0, 1.0),
            ),
            track_point_entities = Property(
                type=List[TrackPointEntity],
                default_value=list(),
            ),
            points = Property(
                type=List[Tuple[float, float]],
                default_value=list(),
            ),
            track_points = Property(
                type=Dict[int, Property[Tuple[float, float]]],
                default_value=dict(),
            ),
        )
        
        self.point_size = point_size
        self.track_color = track_color

    def render(self, frame: Union[float, None], viewer: 'ImageViewerGLWidget'):

        if frame not in self.props.points:
            return

        points = self.props.points[frame]
        gl_points = [viewer.canvas_to_gl_coords(point) for point in points]

        viewer.shader_program.set_color(*self.track_color)

        vbo = draw_points(gl_points, point_size=self.point_size)

        cleanup_vbo(vbo)

    def add_track_points(self, frame, points: List[Tuple[float, float]]):

        id_to_track_point = {
            id(point): Property(type=Tuple[float, float], default_value=(None, None), value=point, frame=frame) for point in points
        }

        self.props.track_points.set_value(id_to_track_point)

        self.props.points[frame].extend(points)

        return id_to_track_point

    def update_track_points(self, frame, update_id_to_position: Dict[int, Tuple[float, float]]):

        id_to_track_point = self.props.track_points.get_value()

        for track_point_id, position in update_id_to_position.items():
            id_to_track_point[track_point_id][frame] = position

        self.props.points[frame].extend(list(update_id_to_position.values()))

    # def add_track_point(self, frame, point: Tuple[float, float]):
    #     self.props.points[frame].append(point)
    #     self.props.gl_points[frame].append(self.viewer.canvas_to_gl_coords(point))

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


    def render(self, frame: Union[float, None], viewer: 'ImageViewerGLWidget'):
        # Ensure blending and depth testing are disabled for 2D rendering
        # GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_DEPTH_TEST)
        
        if self.is_bspline:
            num_points = 100
            curve_points = [self.get_bspline_point(t) for t in np.linspace(self.knot_vector[self.degree], self.knot_vector[-self.degree-1], num_points)]
        else:
            curve_points = self.points

        if self.fill_color:
            self._render_filled_shape(curve_points, viewer.shader_program)

        self._render_outline(curve_points, viewer.shader_program)
        
        if self.show_control_points:
            self._render_control_points(viewer.shader_program)

    def _render_filled_shape(self, curve_points, shader_program):
        vbo = create_vbo(curve_points)

        shader_program.set_color(*self.fill_color)
        
        draw_with_vbo(vbo, GL.GL_POLYGON, len(curve_points))
        cleanup_vbo(vbo)

    def _render_outline(self, curve_points, shader_program):

        vbo = create_vbo(curve_points)
        GL.glLineWidth(self.line_width)

        shader_program.set_color(*self.line_color)

        draw_with_vbo(vbo, GL.GL_LINE_LOOP, len(curve_points))
        cleanup_vbo(vbo)

    def _render_control_points(self, shader_program):

        shader_program.set_color(0.1, 0.5, 0.8, 0.5)
        vbo = draw_points(self.points)

        cleanup_vbo(vbo)

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
    def __init__(self, viewer: 'ImageViewerGLWidget', width: int, height: int):
        super().__init__()
        self.viewer = viewer
        self.width = width
        self.height = height

        self.rectangle_mesh = RectangleMesh()
        self.texture = Texture2D(self.width, self.height)

    def set_size(self, width: int, height: int):
        self.width = width
        self.height = height

    def set_image(self, image: np.ndarray = None):
        self.image = image
        self.height, self.width = self.image.shape[:2]
        self.texture.set_texture_size(self.width, self.height)
        self.texture.set_image(self.image)

    def render(self):
        self.viewer.shader_program.use_texture()

        # Bind the texture to the current active texture unit
        with self.texture:
            GL.glBindVertexArray(self.rectangle_mesh.vao)
            GL.glDrawArrays(GL.GL_QUADS, 0, 4)
            GL.glBindVertexArray(0)
