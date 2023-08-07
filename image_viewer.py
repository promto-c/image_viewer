import sys
import cv2
import numpy as np
from functools import wraps

from typing import Tuple, List

from OpenGL import GL
import OpenGL.GL.shaders as shaders
from PyQt5 import QtCore, QtGui, QtWidgets

from tranformation import apply_transformation, create_translation_matrix, create_rotation_matrix, create_scale_matrix
from entity import Entity, CanvasEntity, LayerEntity, LineEntity

def clamp(value: float, min_value: float, max_value: float) -> float:
    """Helper method to clamp a value between a minimum and maximum value
    """
    return max(min(value, max_value), min_value)

class ImageViewerGLWidget(QtWidgets.QOpenGLWidget):

    MIN_ZOOM = 0.1
    MAX_ZOOM = 10.0

    ZOOM_STEP = 0.1

    # Initialization and Setup
    # ------------------------
    def __init__(self, parent=None, image_data=None):
        # Initialize the super class
        super().__init__(parent)

        # Store the arguments
        self.image_data = image_data

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
        self.image_height, self.image_width, _c = self.image_data.shape

        # test draw line
        self.line_start = None
        self.line_end = None

        self.entities: List[Entity] = list()

        self.shader_program = None
        self.canvas_entity = None

        # Set default values for lift, gamma, and gain
        self.lift = 0.0
        self.gamma = 1.0
        self.gain = 1.0

        # Private Attributes
        # ------------------
        self._drag_start = None
        self._viewer_zoom = 1.0
        self._drag_offset = (0.0, 0.0)
        
    def _setup_ui(self):
        """Set up the UI for the widget, including creating widgets and layouts.
        """
        # 
        pass

    def _setup_signal_connections(self):
        """Set up signal connections between widgets and slots.
        """
        # Connect signals to slots here
        pass

    # Private Methods
    # ---------------
    def _initial_shader_program(self):
        # Create vertex shader
        vertex_shader_source = '''
            #version 330 core
            layout (location = 0) in vec2 position;
            uniform mat4 transformation;  // Uniform variable for transformation matrix

            out vec2 TexCoords;

            void main() {
                vec4 transformed_position = transformation * vec4(position, 0.0, 1.0);  // Apply the transformation
                gl_Position = transformed_position;
                TexCoords = position * 0.5 + 0.5; // map from [-1,1] to [0,1]
            }
        '''

        fragment_shader_source = '''
            #version 330 core
            uniform sampler2D imageTexture;
            uniform float lift;
            uniform float gamma;
            uniform float gain;
            in vec2 TexCoords;

            out vec4 FragColor;

            void main() {
                vec4 color = texture(imageTexture, TexCoords);
                color.rgb = pow(color.rgb * gain + lift, vec3(1.0/gamma));
                FragColor = color;
            }
        '''

        # Compile and link the shader program
        self.shader_program = shaders.compileProgram(
            shaders.compileShader(vertex_shader_source, GL.GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader_source, GL.GL_FRAGMENT_SHADER)
        )

        # Set uniform locations
        self.uniform_loc_transformation = GL.glGetUniformLocation(self.shader_program, "transformation")
        # Set uniform locations, image adjustments
        self.uniform_loc_lift = GL.glGetUniformLocation(self.shader_program, "lift")
        self.uniform_loc_gamma = GL.glGetUniformLocation(self.shader_program, "gamma")
        self.uniform_loc_gain = GL.glGetUniformLocation(self.shader_program, "gain")

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
        GL.glUniform1f(self.uniform_loc_lift, self.lift)
        GL.glUniform1f(self.uniform_loc_gamma, self.gamma)
        GL.glUniform1f(self.uniform_loc_gain, self.gain)

    @_use_shader
    def _update_viewer_transformation(self):
        # Calculate the x and y offsets to center the image
        x_offset = (self._drag_offset[0]/self.width()*2)
        y_offset = -(self._drag_offset[1]/self.height()*2)

        # Calculate the scaled width and height of the image
        scaled_width = (self.image_width/self.width()) * self._viewer_zoom
        scaled_height = (self.image_height/self.height()) * self._viewer_zoom

        translation_matrix = create_translation_matrix(x_offset, y_offset)
        scale_matrix = create_scale_matrix(scaled_width, scaled_height)

        self.viewer_transformation_matrix = np.dot(translation_matrix, scale_matrix)

        # Update the uniform variable in the shader
        GL.glUniformMatrix4fv(self.uniform_loc_transformation, 1, GL.GL_FALSE, self.viewer_transformation_matrix.T.flatten())

    @_use_shader
    def _render(self):
        # Render the entities
        self.canvas_entity.render()
        for entity in self.entities:
            entity.render()

    # Extended Methods
    # ----------------
    def set_image(self, image_data: np.ndarray) -> None:
        """Set the image to be displayed.

        Args:
            image_data (np.ndarray): Image data as a NumPy array.
        """
        self.image_data = image_data
        self.canvas_entity.set_image(image_data)

        self.update()

    def pixel_to_gl_coords(self, pixel_coords: QtCore.QPoint) -> Tuple[float, float]:
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

    def fit_in_view(self):
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
    def initializeGL(self):
        """Initialize the OpenGL context.
        """
        # Set the clear color for the OpenGL context
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)

        self._initial_shader_program()
        self.canvas_entity = CanvasEntity()

        # Set the image to display
        self.set_image(self.image_data)

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
            self.line_start = self.pixel_to_gl_coords(event.pos())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse release event.

        Args:
            event (QtGui.QMouseEvent): Mouse event.
        """
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.line_end = self.pixel_to_gl_coords(event.pos())

            line = LineEntity(line_start=self.line_start, line_end=self.line_end)
            self.entities.append(line)

            self.update()

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

        # Store the previous zoom level for comparison
        prev_viewer_zoom = self._viewer_zoom

        # Update the zoom level and clamp it within the minimum and maximum zoom factors
        self._viewer_zoom = clamp(self._viewer_zoom + zoom_delta, self.MIN_ZOOM, self.MAX_ZOOM)

        # Only adjust the drag offset if the zoom level has changed
        if self._viewer_zoom != prev_viewer_zoom:
            # Compute the actual change in zoom level
            zoom_delta = self._viewer_zoom - prev_viewer_zoom

            # Compute the scaled width and height of the image based on the zoom level change
            scaled_width = (self.image_width/self.width()) * zoom_delta
            scaled_height = (self.image_height/self.height()) * zoom_delta

            # Convert the cursor position from pixel coordinates to OpenGL coordinates
            gl_pos_x, gl_pos_y = self.pixel_to_gl_coords(event.pos())

            # Scale the cursor position by the scaled width and height
            gl_pos_x *= scaled_width
            gl_pos_y *= scaled_height

            # Compute the translation in the x and y directions
            translate_x = - gl_pos_x * self.width()/2
            translate_y = gl_pos_y * self.height()/2

            # Update the drag offset with this translation
            self._drag_offset = (self._drag_offset[0] + translate_x, self._drag_offset[1] + translate_y)

        # Redraw the widget to apply the zoom transformation
        self.update()

class MainUI(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        image_path = r'example_image.1001.jpg'
        image_path2 = r'example_image.1002.jpg'

        image_data = cv2.imread(image_path)
        self.image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        # self.image = self.image.astype(np.float32) / 255.0

        image_data2 = cv2.imread(image_path2)
        self.image_data2 = cv2.cvtColor(image_data2, cv2.COLOR_BGR2RGB)
        self.image_data2 = self.image_data2.astype(np.float32) / 255.0

        self.setup_ui()

    def setup_ui(self):
        self.gl_widget = ImageViewerGLWidget(self, image_data=self.image_data)
        self.switch_button = QtWidgets.QPushButton("Switch Image", self)
        self.switch_button.clicked.connect(self.switch_image)

        self.main_layout = QtWidgets.QGridLayout()
        self.main_layout.addWidget(self.gl_widget, 0, 0)
        self.main_layout.addWidget(self.switch_button, 1, 0)
        self.setLayout(self.main_layout)

    def switch_image(self):
        if np.array_equal(self.gl_widget.image_data, self.image_data):
            self.gl_widget.set_image(self.image_data2)
        else:
            self.gl_widget.set_image(self.image_data)

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    win = MainUI()
    win.show()
    sys.exit( app.exec() )
