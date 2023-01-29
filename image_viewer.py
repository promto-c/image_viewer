import sys
import cv2
import numpy as np

from OpenGL import GL
from PyQt5 import QtCore, QtGui, QtWidgets

class GLWidget( QtWidgets.QOpenGLWidget ):
    ''' A class to display an image using OpenGL
    '''

    def __init__(self, parent=None, image=None):
        ''' Constructor
        '''
        # Initialize the base class
        super(GLWidget, self).__init__(parent)
        # QtWidgets.QOpenGLWidget

        # Set the image and its dimensions
        self.image = image

        self.image_height, self.image_width, _c = self.image.shape

        # Initialize the zoom level and step
        self.zoom = 1.0
        self.zoom_step = 0.1

        # Initialize the drag offset and start point
        self.drag_offset = (0.0, 0.0)
        self.drag_start = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
    def sizeHint(self):
        ''' Override the sizeHint method to set the preferred size of the widget
        '''
        return QtCore.QSize(self.image_width, self.image_height)
 
    def set_image(self, image_data):
        ''' Method to set the image to display
        '''
        # Get the height and width of the image
        height, width, _channel = image_data.shape

        # Flip the image vertically
        image_data = np.flipud(image_data)

        # Check the data type of the image
        if image_data.dtype == np.uint8:
            pixel_data_type = GL.GL_UNSIGNED_BYTE
            texture_format = GL.GL_RGB
        elif image_data.dtype == np.float32:
            pixel_data_type = GL.GL_FLOAT
            texture_format = GL.GL_RGB32F
        else:
            return

        
        # Use glTexImage2D to set the image
        self.gl.glTexImage2D(
            self.gl.GL_TEXTURE_2D, 
            0, 
            texture_format, 
            width, height,
            0,
            self.gl.GL_RGB, 
            pixel_data_type, 
            image_data.flatten().tolist() )
 
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        ''' Method to handle mouse press events
        '''
        # If the middle button is pressed, set the drag start point
        if event.button() == QtCore.Qt.MiddleButton:
            self.drag_start = (event.x(), event.y())
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        ''' Method to handle mouse move events
        '''
        # If the middle button is pressed and the drag start point is set
        if event.buttons() & QtCore.Qt.MiddleButton and self.drag_start:
            # Calculate the x and y offsets
            x_offset = event.x() - self.drag_start[0]
            y_offset = self.drag_start[1] - event.y()
            self.drag_offset = (x_offset, y_offset)

            # Update the widget to apply the drag transformation
            self.update()
    
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        ''' Method to handle wheel events
        '''
        # Check the scroll direction and update the zoom level accordingly
        if event.angleDelta().y() > 0:
            self.zoom += self.zoom_step
        else:
            self.zoom -= self.zoom_step

        # Update the widget to apply the zoom transformation
        self.update()

    def fit_image_in_view(self):
        ''' Method to fit the image in the view without changing the image size ratio
        '''
        # Calculate the drag offset to center the image in the view
        self.drag_offset = (0.0, 0.0)
        image_aspect_ratio = self.image_width / self.image_height
        view_aspect_ratio = self.width() / self.height()
        if image_aspect_ratio > view_aspect_ratio:
            self.drag_offset = ((self.width() - self.image_width * self.zoom) / 2, 0.0)
        else:
            self.drag_offset = (0.0, (self.height() - self.image_height * self.zoom) / 2)
            
        # Update the widget to apply the resize transformations
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        ''' Method to handle resize events
        '''
        # Call the method to fit the image in the view
        self.fit_image_in_view()

    def initializeGL(self):
        ''' Method to initialize the OpenGL context
        '''
        # Create a version profile and set the version to 2.0
        version_profile = QtGui.QOpenGLVersionProfile()
        version_profile.setVersion(2,0)
        # Get the functions for the version profile
        self.gl: GL = self.context().versionFunctions(version_profile)
        # Set the clear color to black
        self.gl.glClearColor(0.0, 0.0, 0.0, 1.0) 
        # Set the image to display
        self.set_image( self.image )        

    def resizeGL(self, width: int, height: int) -> None:
        ''' Method to handle resizing of the widget
        '''
        # Set the viewport to the new width and height of the widget
        self.gl.glViewport(0, 0, width, height)

    def paintGL(self):
        ''' Method to paint the widget. This method is called by the system whenever the widget needs to be repainted. 
        '''
        # Clear the buffer
        self.gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Set the viewport
        self.gl.glViewport(0, 0, self.image_width, self.image_height)

        # Enable texture mapping
        self.gl.glEnable(GL.GL_TEXTURE_2D)

        # Sets the texture minification/magnification filter
        self.gl.glTexParameterf(
            GL.GL_TEXTURE_2D,          # target
            GL.GL_TEXTURE_MIN_FILTER,  # pname
            GL.GL_NEAREST              # param
            )
        self.gl.glTexParameterf(
            GL.GL_TEXTURE_2D,          # target
            GL.GL_TEXTURE_MAG_FILTER,  # pname
            GL.GL_NEAREST              # param
            )

        # Set the projection matrix
        self.gl.glMatrixMode(GL.GL_PROJECTION)
        self.gl.glLoadIdentity()
        self.gl.glOrtho(0, self.image_width, 0, self.image_height,-1,1)

        # Set the modelview matrix
        self.gl.glMatrixMode(GL.GL_MODELVIEW)
        self.gl.glLoadIdentity()

        # Apply the zoom and drag transformations
        self.gl.glScaled(self.zoom, self.zoom, 1.0)
        self.gl.glTranslated(self.drag_offset[0]/self.zoom, self.drag_offset[1]/self.zoom, 0.0)

        # Draw the image quad
        self.gl.glBegin(GL.GL_QUADS)
        self.gl.glTexCoord2f(0.0, 0.0)
        self.gl.glVertex2f(0.0, 0.0)
        self.gl.glTexCoord2f(1.0, 0.0)
        self.gl.glVertex2f(self.image_width, 0.0)
        self.gl.glTexCoord2f(1.0, 1.0)
        self.gl.glVertex2f(self.image_width, self.image_height)
        self.gl.glTexCoord2f(0.0, 1.0)
        self.gl.glVertex2f(0.0, self.image_height)
        self.gl.glEnd()
        
        # Flush the OpenGL pipeline to ensure that all commands are executed
        self.gl.glFlush()

class MainUI(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(MainUI, self).__init__(parent)

        image_path = r'example_image.jpg'

        image = cv2.imread( image_path )
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # NOTE: Tets float image
        self.image = self.image.astype( np.float32 ) / 255.0

        self.setup_ui()

    def setup_ui(self):
        self.gl_widget = GLWidget(self, image=self.image)
        self.main_layout = QtWidgets.QGridLayout()
        self.main_layout.addWidget(self.gl_widget,0,0)
        self.setLayout(self.main_layout)

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    win = MainUI()
    win.show()
    sys.exit( app.exec() )