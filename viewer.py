# Standard Library Imports
# ------------------------
import sys
import numpy as np
from functools import wraps
from numbers import Number
from typing import Callable, Tuple, List, Union

# Third Party Imports
# -------------------
import cv2
from OpenGL import GL
from qtpy import QtCore, QtGui, QtWidgets

# Local Imports
# -------------
from theme import theme
from tranformation import apply_transformation
from entity import Entity, CanvasEntity, LayerEntity, ShapeEntity
from shaders.viewer_shader import ViewerShaderProgram
from utils.image_utils import ImageSequence
from nodes.node import Node


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Helper method to clamp a value between a minimum and maximum value
    """
    return max(min(value, max_value), min_value)

class ImageViewerGLWidget(QtWidgets.QOpenGLWidget):

    MIN_ZOOM = 0.1
    MAX_ZOOM = 10.0

    ZOOM_STEP = 0.1

    left_mouse_pressed = QtCore.Signal(QtGui.QMouseEvent)
    left_mouse_released = QtCore.Signal(QtGui.QMouseEvent)
    left_mouse_moved = QtCore.Signal(QtGui.QMouseEvent)

    # Initialization and Setup
    # ------------------------
    def __init__(self, image: Union[Tuple[int, int], np.ndarray, ImageSequence] = None, parent=None):
        # Initialize the super class
        super().__init__(parent)

        format = QtGui.QSurfaceFormat()
        format.setSamples(4)  # Change the number for more or less anti-aliasing
        QtGui.QSurfaceFormat.setDefaultFormat(format)

        # Store the arguments
        self.image = image

        # Set up the initial values
        self._setup_attributes()
        # Set up signal connections
        self._setup_signal_connections()

    def _setup_attributes(self):
        """Set up the initial values for the widget.
        """
        # Attributes
        # ----------
        self.vector_entities: List[Entity] = list()

        self.shader_program = None
        self.canvas_entity = None

        self.current_frame = 0.0

        # Set default values for lift, gamma, and gain
        self.lift = 0.0
        self.gamma = 1.0
        self.gain = 1.0

        self.pixel_aspect_ratio = 1.0

        # Private Attributes
        # ------------------
        self._drag_start = None
        self._viewer_zoom = 1.0
        self._drag_offset = (0.0, 0.0)

        self.set_image(self.image)

    # TODO: Refactor
    def set_image(self, image: Union[np.ndarray, Tuple[int, int], ImageSequence, Node]) -> None:
        """Set the image to be displayed.

        Args:
            image (np.ndarray): Image data as a NumPy array.
        """
        self.image = image

        if isinstance(self.image, tuple):
            self.canvas_width, self.canvas_height = self.image
            self.image = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.float32)
            image_data = self.image

        elif isinstance(self.image, np.ndarray):
            self.canvas_height, self.canvas_width, _c = self.image.shape
            image_data = self.image

        elif isinstance(self.image, ImageSequence):
            image_data = self.image.get_image_data(self.current_frame)
            self.canvas_height, self.canvas_width, _c = image_data.shape

        elif isinstance(self.image, Node):
            image_data = self.image.get_image_data(self.current_frame)
            if image_data is None:
                self.canvas_width, self.canvas_height = 100, 100
                image_data = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.float32)
            else:
                self.canvas_height, self.canvas_width, _c = image_data.shape

        else:
            self.canvas_width, self.canvas_height = 100, 100
            self.image = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.float32)
            image_data = self.image

        if self.canvas_entity is None:
            return

        self.canvas_entity.set_image(image_data)
        self.update()

    def _setup_signal_connections(self):
        """Set up signal connections between widgets and slots.
        """
        # Key Binds
        # --------
        self.key_bind('F', self.fit_in_view)

    # Private Methods
    # ---------------
    def _use_shader(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.shader_program:
                result = func(self, *args, **kwargs)
            return result
        return wrapper

    @_use_shader
    def _update_viewer_adjustments(self):
        # Set the uniform values for image adjustments
        self.shader_program.set_lift(self.lift)
        self.shader_program.set_gamma(self.gamma)
        self.shader_program.set_gain(self.gain)

    @_use_shader
    def _update_viewer_transformation(self):
        # Calculate the x and y offsets to center the image
        x_offset = (self._drag_offset[0] / self.width()) * 2
        y_offset = -(self._drag_offset[1] / self.height()) * 2

        # Calculate the scaled width and height of the image, adjusting for pixel aspect ratio
        scaled_width = (self.canvas_width / self.width()) * self._viewer_zoom * self.pixel_aspect_ratio
        scaled_height = (self.canvas_height / self.height()) * self._viewer_zoom

        # Compute the required transformation matrix
        self.viewer_transformation_matrix = np.array([
            [scaled_width, 0, 0, x_offset],
            [0, scaled_height, 0, y_offset],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Update the shader with the new transformation
        self.shader_program.set_tranformation(self.viewer_transformation_matrix)

    @_use_shader
    def _render(self):
        # Render the canvas entity
        self.canvas_entity.render()

        # Render the vector entities
        for entity in self.vector_entities:
            entity.render(self.current_frame, self)

        if isinstance(self.image, Node):
            for entity in self.image.vector_entities:
                entity.render(self.current_frame, self)

    def _handle_viewer_zoom(self, zoom_delta: float, pivot_coords: QtCore.QPoint) -> None:
        """Handle the viewer zoom based on the zoom delta and wheel event

        Args:
            zoom_delta(float): The zoom delta.
            pivot_coords (QtCore.QPoint): The pivot coordinates for the zoom.
        """
        # Store the previous zoom level for comparison
        prev_viewer_zoom = self._viewer_zoom

        # Update the zoom level and clamp it within the minimum and maximum zoom factors
        self._viewer_zoom = clamp(self._viewer_zoom + zoom_delta, self.MIN_ZOOM, self.MAX_ZOOM)

        # Only adjust the drag offset if the zoom level has changed
        if self._viewer_zoom != prev_viewer_zoom:
            # Compute the actual change in zoom level
            zoom_delta = self._viewer_zoom - prev_viewer_zoom

            # Compute the scaled width and height of the image based on the zoom level change
            scaled_width = (self.canvas_width/self.width()) * zoom_delta
            scaled_height = (self.canvas_height/self.height()) * zoom_delta

            # Convert the cursor position from pixel coordinates to OpenGL coordinates
            gl_pos_x, gl_pos_y = self.display_to_gl_coords(pivot_coords)

            # Scale the cursor position by the scaled width and height
            gl_pos_x *= scaled_width
            gl_pos_y *= scaled_height

            # Compute the translation in the x and y directions
            translate_x = - gl_pos_x * self.width() / 2
            translate_y = gl_pos_y * self.height() / 2

            # Update the drag offset with this translation
            self._drag_offset = (self._drag_offset[0] + translate_x, self._drag_offset[1] + translate_y)

    # Extended Methods
    # ----------------
    def set_frame(self, frame: Number):
        if not isinstance(self.image, (ImageSequence, Node)):
            # TODO: Print something
            return

        self.current_frame = frame
        self.canvas_entity.set_image(self.image.get_image_data(frame))
        self.update()

    def set_pixel_aspect_ratio(self, par: float):
        """Set the pixel aspect ratio for the image display.

        Args:
            par (float): The new pixel aspect ratio.
        """
        self.pixel_aspect_ratio = par
        self.update()

    # def set_canvas_size(self, width: int, height: int):
    #     self.canvas_width, self.canvas_height = width, height

    def display_to_gl_coords(self, pixel_coords: QtCore.QPoint) -> Tuple[float, float]:
        """Convert pixel coordinates to OpenGL coordinates.

        Args:
            pixel_coords (QtCore.QPoint): Pixel coordinates.

        Returns:
            Tuple[float, float]: OpenGL coordinates.
        """
        # Normalize coordinates to [0, 1] range
        x = pixel_coords.x() / self.width()
        y = pixel_coords.y() / self.height()

        # Shift coordinates to [-1, 1] range
        x = 2 * x - 1
        y = 2 * (1 - y) - 1

        # Apply the transformation matrix
        x, y = apply_transformation((x, y), np.linalg.inv(self.viewer_transformation_matrix))

        return x, y
    
    def canvas_to_gl_coords(self, pixel_coords: Tuple[float, float]) -> Tuple[float, float]:
        # Normalize coordinates to [0, 1] range based on the image size
        x = pixel_coords[0] / self.canvas_width
        y = pixel_coords[1] / self.canvas_height

        # Shift coordinates to [-1, 1] range
        x = 2 * x - 1
        y = 2 * (1 - y) - 1

        return x, y

    def gl_to_canvas_coords(self, gl_coords: Tuple[float, float]) -> Tuple[float, float]:
        x, y = gl_coords

        x = (x + 1) / 2
        y = 1 - ((y + 1) / 2)
        
        x = x * self.canvas_width
        y = y * self.canvas_height

        return x, y

    def fit_in_view(self):
        """Fits the image within the widget view while maintaining the aspect ratio.
        """
        # Calculate the new zoom level while preserving the aspect ratio
        self._viewer_zoom = min(self.width() / self.canvas_width, self.height() / self.canvas_height)

        # Reset the drag offset
        self._drag_offset = (0.0, 0.0)

        # Update the widget to reset the zoom level and offset
        self.update()

    def key_bind(self, key_sequence: str, function: Callable):
        # Create a shortcut
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key_sequence), self)
        # Connect the activated signal of the shortcut to the slot
        shortcut.activated.connect(function)

    # OpenGL Initialization and Setup
    # -------------------------------
    def initializeGL(self):
        """Initialize the OpenGL context.
        """
        # Enable multi-sampling for smoother rendering
        GL.glEnable(GL.GL_MULTISAMPLE)

        # Set the clear color for the OpenGL context
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)

        # Initialize the shader program and canvas entity.
        self.shader_program = ViewerShaderProgram(self)
        self.canvas_entity = CanvasEntity(self, self.canvas_width, self.canvas_height)

        # Adjust the view to fit the content.
        self.fit_in_view()

    def paintGL(self) -> None:
        """Paint the OpenGL widget.
        """
        self._update_viewer_transformation()
        self._update_viewer_adjustments()

        self._render()

        # Flush the OpenGL pipeline to ensure that all commands are executed
        GL.glFlush()

    # Event Handling or Override Methods
    # ----------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press event.

        Args:
            event (QtGui.QMouseEvent): Mouse event.
        """
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._drag_start = (event.x(), event.y())
            self._prev_drag_offset = self._drag_offset
    
        elif event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.left_mouse_pressed.emit(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release event.

        Args:
            event (QtGui.QMouseEvent): Mouse event.
        """
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.left_mouse_released.emit(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move event.

        Args:
            event (QtGui.QMouseEvent): Mouse event.
        """
        if event.buttons() & QtCore.Qt.MouseButton.MiddleButton and self._drag_start:
            x_offset = event.x() - self._drag_start[0]
            y_offset = event.y() - self._drag_start[1]
            self._drag_offset = (self._prev_drag_offset[0] + x_offset, self._prev_drag_offset[1] + y_offset)
            self.update()

        if event.buttons() & QtCore.Qt.MouseButton.MiddleButton:
            self.left_mouse_moved.emit(event)

    def sizeHint(self) -> QtCore.QSize:
        """Get the preferred size of the widget.

        Returns:
            QtCore.QSize: Preferred size of the widget as a QSize.
        """
        return QtCore.QSize(self.canvas_width, self.canvas_height)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Method to handle wheel events

        Args:
            event(QtGui.QWheelEvent): Wheel event.
        """
        # Calculate the zoom delta based on the scroll direction
        zoom_delta = self.ZOOM_STEP if event.angleDelta().y() > 0 else -self.ZOOM_STEP

        # Handle the viewer zoom
        self._handle_viewer_zoom(zoom_delta, event.pos())

        # Redraw the widget to apply the zoom transformation
        self.update()

class MainUI(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        image_path = r'example_image.1001.jpg'

        image = cv2.imread(image_path)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self._setup_ui()
        self._setup_signal_connections()

    def _setup_ui(self):
        self.viewer = ImageViewerGLWidget(self.image, self)
        self.current = self.image

        self.main_layout = QtWidgets.QGridLayout()
        self.main_layout.addWidget(self.viewer, 0, 0)
        self.setLayout(self.main_layout)

        self.shape = ShapeEntity(points=[(0.0, 0.0)])

        self.viewer.vector_entities.append(self.shape)

    def _setup_signal_connections(self):
        self.viewer.left_mouse_pressed.connect(self.store_start_pos)

    def store_start_pos(self, event: QtGui.QMouseEvent):
        self.line_start = self.viewer.display_to_gl_coords(event.pos())

        self.shape.add_point(self.line_start)
        self.viewer.update()

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    theme.set_theme(app, theme='dark')
    win = MainUI()
    win.show()
    sys.exit( app.exec() )
