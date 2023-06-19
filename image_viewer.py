import sys
import cv2
import numpy as np

from typing import Tuple, List

from OpenGL import GL
from PyQt5 import QtCore, QtGui, QtWidgets

def apply_transformation(positions, transformation_matrix):
    # Apply the transformation matrix to the vertex positions using a list comprehension
    return [np.dot(transformation_matrix, (v[0], v[1], 0.0, 1.0))[:2] for v in positions]

def create_translation_matrix(x_translation: float, y_translation: float, z_translation: float = 0.0) -> np.ndarray:
    """Creates a 3D translation matrix based on the provided translation values.

    Args:
        x_translation (float): The translation along the x-axis.
        y_translation (float): The translation along the y-axis.
        z_translation (float, optional): The translation along the z-axis. Defaults to 0.0.

    Returns:
        np.ndarray: The resulting translation matrix as a 4x4 NumPy array.
    """
    translation_matrix = np.array([[1.0, 0.0, 0.0, x_translation],
                                   [0.0, 1.0, 0.0, y_translation],
                                   [0.0, 0.0, 1.0, z_translation],
                                   [0.0, 0.0, 0.0, 1.0]])
    return translation_matrix

def create_rotation_matrix(angle: float = 0.0, axis: List[float] = [0.0, 0.0, 1.0]) -> np.ndarray:
    """Creates a 3D rotation matrix based on the provided angle and axis.

    Args:
        angle (float, optional): The rotation angle in radians. Defaults to 0.0.
        axis (List[float], optional): The rotation axis as a 3D vector. Defaults to [0.0, 0.0, 1.0].

    Returns:
        ndarray: The resulting rotation matrix as a 4x4 NumPy array.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)

    ux, uy, uz = axis
    rotation_matrix = np.array([[cos_theta + ux ** 2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta, 0.0],
                                [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy ** 2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta, 0.0],
                                [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz ** 2 * (1 - cos_theta), 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
    return rotation_matrix

class Entity:
    def __init__(self, texture_id=None, width=0, height=0, line_start=None, line_end=None):
        self.texture_id = texture_id
        self.width = width
        self.height = height
        self.line_start = line_start
        self.line_end = line_end

    def render(self):
        raise NotImplementedError("Subclasses must implement the render method.")

class LayerEntity(Entity):
    def __init__(self, transformation_matrix=np.eye(4)):
        self.transformation_matrix = transformation_matrix
        self.children = []

    def render(self):
        # Apply the transformation matrix to the children
        for child in self.children:
            child.render(self.transformation_matrix)

    def add_child(self, child):
        self.children.append(child)

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)

    def clear_children(self):
        self.children = []

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

class ImageEntity(Entity):
    def __init__(self, texture_id, width, height):
        self.texture_id = texture_id
        self.width = width
        self.height = height

    def render(self, transformation_matrix=np.eye(4)):
        # Bind the texture to the current active texture unit
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

        # Define the vertex positions
        vertex_positions = self.get_vertex_positions()

        # Apply the transformation matrix to the vertex positions
        transformed_positions = apply_transformation(vertex_positions, transformation_matrix)

        # Draw the image quad
        self.draw_textured_quad(transformed_positions)

    def get_vertex_positions(self):
        # Define the vertex positions
        return [
            (0.0, 0.0),
            (self.width, 0.0),
            (self.width, self.height),
            (0.0, self.height)
        ]

    def draw_textured_quad(self, positions):
        # Set the texture coordinates and vertex positions for the quad
        tex_coords = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]

        # Draw the image quad
        GL.glBegin(GL.GL_QUADS)
        for tex_coord, position in zip(tex_coords, positions):
            GL.glTexCoord2f(*tex_coord)
            GL.glVertex2f(*position)
        GL.glEnd()

class ImageViewerGLWidget(QtWidgets.QOpenGLWidget):

    MIN_ZOOM = 0.1
    MAX_ZOOM = 10.0

    ZOOM_STEP = 0.1

    # Initialization and Setup
    # ------------------------
    def __init__(self, parent=None, image=None):
        # Initialize the super class
        super().__init__(parent)

        # Store the arguments
        self.image = image

        # Set up the initial values
        self._setup_initial_values()
        # Set up the UI
        self._setup_ui()
        # Set up signal connections
        self._setup_signal_connections()

    def _setup_initial_values(self):
        """Set up the initial values for the widget.
        """
        # Attributes
        # ------------------
        self.image_height, self.image_width, _c = self.image.shape

        self.texture_id = None

        # test draw line
        self.line_start = None
        self.line_end = None

        self.entities: List[Entity] = list()

        # Private Attributes
        # ------------------
        self._viewer_zoom = 1.0

        self._drag_offset = (0.0, 0.0)
        self._drag_start = None

    def _setup_ui(self):
        """Set up the UI for the widget, including creating widgets and layouts.
        """
        # 
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def _setup_signal_connections(self):
        """Set up signal connections between widgets and slots.
        """
        # Connect signals to slots here
        pass

    # Private Methods
    # ---------------

    # Extended Methods
    # ----------------
    def set_image(self, image_data: np.ndarray) -> None:
        """Set the image to be displayed.

        Args:
            image_data (np.ndarray): Image data as a NumPy array.
        """
        self.image = image_data
        # Get the height and width of the image
        height, width, _channel = self.image.shape
        
        # Flip the image vertically using OpenCV's flip function
        image_data = cv2.flip(self.image, 0)

        # Check the data type of the image and set the corresponding pixel data type and texture format for OpenGL
        if image_data.dtype == np.uint8:
            pixel_data_type = GL.GL_UNSIGNED_BYTE
            texture_format = GL.GL_RGB
        elif image_data.dtype == np.float32:
            pixel_data_type = GL.GL_FLOAT
            texture_format = GL.GL_RGB32F
        else:
            # Return early if the data type is not supported
            return

        # Create an OpenGL texture ID and bind the texture
        self.texture_id = GL.glGenTextures(1)

        #
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

        # Set the texture minification/magnification filter
        GL.glTexParameterf(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MIN_FILTER,
            GL.GL_NEAREST
        )
        GL.glTexParameterf(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MAG_FILTER,
            GL.GL_NEAREST
        )
        
        # Use glTexImage2D to set the image texture in OpenGL
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,          # target
            0,                              # level
            texture_format,                 # internal format
            width, height,                  # width and height of the texture
            0,                              # border (must be 0)
            GL.GL_RGB,                 # format of the pixel data
            pixel_data_type,                # data type of the pixel data
            image_data.flatten().tolist()   # flattened image data as a list
        )

        # Create an instance of ImageEntity with the texture ID, width, and height
        image_entity = ImageEntity(self.texture_id, self.image_width, self.image_height)
        self.entities.append(image_entity)

        self.update()

    def pixel_to_gl_coords(self, pixel_coords: QtCore.QPoint) -> Tuple[float, float]:
        """Convert pixel coordinates to OpenGL coordinates.

        Args:
            pixel_coords (QtCore.QPoint): Pixel coordinates.

        Returns:
            Tuple[float, float]: OpenGL coordinates.
        """
        # Calculate the scaled width and height of the image
        scaled_width = self.image_width * self._viewer_zoom
        scaled_height = self.image_height * self._viewer_zoom

        # Calculate the x and y offsets to center the image
        x_offset = (self.width() - scaled_width) / 2 + self._drag_offset[0]
        y_offset = (self.height() - scaled_height) / 2 - self._drag_offset[1]

        # Calculate the x and y coordinates in GL space
        x = (pixel_coords.x() - x_offset) / self._viewer_zoom
        y = (self.height() - pixel_coords.y() - y_offset) / self._viewer_zoom

        # Flip the y-coordinate
        y = self.image_height - y

        return x, y

    @staticmethod
    def clamp(value: float, min_value: float, max_value: float) -> float:
        """Helper method to clamp a value between a minimum and maximum value
        """
        return max(min(value, max_value), min_value)

    def fit_image_in_view(self):
        """Fits the image within the widget view while maintaining the aspect ratio.
        """
        # Calculate the new zoom level while preserving the aspect ratio
        self._viewer_zoom = min(self.width() / self.image_width, self.height() / self.image_height)

        # Reset the drag offset
        self._drag_offset = (0.0, 0.0)

        # Update the widget to reset the zoom level and offset
        self.update()

    # OpenGL Initialization and Setup
    # -------------------------------
    def resizeGL(self, width: int, height: int) -> None:
        """Method to handle resizing of the widget

        Args:
            width: The new width of the widget.
            height: The new height of the widget.
        """
        # Set the viewport to the new width and height of the widget
        GL.glViewport(0, 0, width, height)

    def initializeGL(self):
        """Initialize the OpenGL context.
        """
        # Set the clear color for the OpenGL context
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

        # Enable texture mapping
        GL.glEnable(GL.GL_TEXTURE_2D)

        # Set the image to display
        self.set_image(self.image)

    def paintGL(self) -> None:
        """Paint the OpenGL widget.
        """
        # Clear the buffer
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, self.width(), self.height(), 0, -1, 1)

        # Set the modelview matrix
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        # Calculate the scaled width and height of the image
        scaled_width = self.image_width * self._viewer_zoom
        scaled_height = self.image_height * self._viewer_zoom

        # Calculate the x and y offsets to center the image
        x_offset = (self.width() - scaled_width) / 2 + self._drag_offset[0]
        y_offset = (self.height() - scaled_height) / 2 + self._drag_offset[1]

        # Apply the translation and scaling transformations
        GL.glTranslatef(x_offset, y_offset, 0.0)
        GL.glScalef(self._viewer_zoom, self._viewer_zoom, 1.0)

        # Render the entities
        if self.entities:
            for entity in self.entities:
                entity.render()

        # Flush the OpenGL pipeline to ensure that all commands are executed
        GL.glFlush()

    # Event Handling or Override Methods
    # ----------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press event.

        Args:
            event (QtGui.QMouseEvent): Mouse event.
        """
        if event.button() == QtCore.Qt.MiddleButton:
            self._drag_start = (event.x(), event.y())
            self._prev__drag_offset = self._drag_offset
    
        elif event.button() == QtCore.Qt.LeftButton:
            self.line_start = self.pixel_to_gl_coords(event.pos())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release event.

        Args:
            event (QtGui.QMouseEvent): Mouse event.
        """
        if event.button() == QtCore.Qt.LeftButton:
            self.line_end = self.pixel_to_gl_coords(event.pos())

            line = LineEntity(line_start=self.line_start, line_end=self.line_end)
            self.entities.append(line)

            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move event.

        Args:
            event (QtGui.QMouseEvent): Mouse event.
        """
        if event.buttons() & QtCore.Qt.MiddleButton and self._drag_start:
            x_offset = event.x() - self._drag_start[0]
            y_offset = event.y() - self._drag_start[1]
            self._drag_offset = (self._prev__drag_offset[0] + x_offset, self._prev__drag_offset[1] + y_offset)
            self.update()

    def sizeHint(self) -> QtCore.QSize:
        """Get the preferred size of the widget.

        Returns:
            QtCore.QSize: Preferred size of the widget as a QSize.
        """
        return QtCore.QSize(self.image_width, self.image_height)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Method to handle wheel events

        Args:
            event(QtGui.QWheelEvent): Wheel event.
        """
        # Calculate the zoom delta based on the scroll direction
        zoom_delta = self.ZOOM_STEP if event.angleDelta().y() > 0 else -self.ZOOM_STEP

        # Update the zoom level and clamp it within the minimum and maximum zoom factors
        self._viewer_zoom = self.clamp(self._viewer_zoom + zoom_delta, self.MIN_ZOOM, self.MAX_ZOOM)

        # Update the widget to apply the zoom transformation
        self.update()

class MainUI(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        image_path = r'example_image.1001.jpg'
        image_path2 = r'example_image.1002.jpg'

        image = cv2.imread(image_path)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # self.image = self.image.astype(np.float32) / 255.0

        image2 = cv2.imread(image_path2)
        self.image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        self.image2 = self.image2.astype(np.float32) / 255.0

        self.setup_ui()

    def setup_ui(self):
        self.gl_widget = ImageViewerGLWidget(self, image=self.image)
        self.switch_button = QtWidgets.QPushButton("Switch Image", self)
        self.switch_button.clicked.connect(self.switch_image)

        self.main_layout = QtWidgets.QGridLayout()
        self.main_layout.addWidget(self.gl_widget, 0, 0)
        self.main_layout.addWidget(self.switch_button, 1, 0)
        self.setLayout(self.main_layout)

    def switch_image(self):
        if np.array_equal(self.gl_widget.image, self.image):
            self.gl_widget.set_image(self.image2)
        else:
            self.gl_widget.set_image(self.image)

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    win = MainUI()
    win.show()
    sys.exit( app.exec() )
