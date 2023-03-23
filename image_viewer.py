import sys
import cv2
import numpy as np

from OpenGL import GL
from PyQt5 import QtCore, QtGui, QtWidgets


class GLWidget(QtWidgets.QOpenGLWidget):
    def __init__(self, parent=None, image=None):
        super(GLWidget, self).__init__(parent)

        self.image = image
        self.image_height, self.image_width, _c = self.image.shape

        self.zoom = 1.0
        self.zoom_step = 0.1

        self.drag_offset = (0.0, 0.0)
        self.drag_start = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def sizeHint(self):
        return QtCore.QSize(self.image_width, self.image_height)

    def set_image(self, image_data):
        height, width, _channel = image_data.shape
        image_data = cv2.flip(image_data, 0)

        if image_data.dtype == np.uint8:
            pixel_data_type = GL.GL_UNSIGNED_BYTE
            texture_format = GL.GL_RGB
        elif image_data.dtype == np.float32:
            pixel_data_type = GL.GL_FLOAT
            texture_format = GL.GL_RGB32F
        else:
            return

        self.gl.glTexImage2D(
            self.gl.GL_TEXTURE_2D,
            0,
            texture_format,
            width, height,
            0,
            self.gl.GL_RGB,
            pixel_data_type,
            image_data.flatten().tolist())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            self.drag_start = (event.x(), event.y())
            self.prev_drag_offset = self.drag_offset

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.buttons() & QtCore.Qt.MiddleButton and self.drag_start:
            x_offset = event.x() - self.drag_start[0]
            y_offset = event.y() - self.drag_start[1]
            self.drag_offset = (self.prev_drag_offset[0] + x_offset, self.prev_drag_offset[1] + y_offset)
            self.update()


    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            self.zoom += self.zoom_step
        else:
            self.zoom -= self.zoom_step
        self.update()

    def fit_image_in_view(self):
        self.drag_offset = (0.0, 0.0)
        self.zoom = min(self.width() / self.image_width, self.height() / self.image_height)
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self.fit_image_in_view()

    def initializeGL(self):
        version_profile = QtGui.QOpenGLVersionProfile()
        version_profile.setVersion(2, 0)
        self.gl: GL = self.context().versionFunctions(version_profile)
        self.gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.set_image(self.image)

    def resizeGL(self, width: int, height: int) -> None:
        self.gl.glViewport(0, 0, width, height)

    def paintGL(self):
        self.gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self.gl.glViewport(0, 0, self.width(), self.height())

        self.gl.glEnable(GL.GL_TEXTURE_2D)

        self.gl.glTexParameterf(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MIN_FILTER,
            GL.GL_NEAREST
        )
        self.gl.glTexParameterf(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MAG_FILTER,
            GL.GL_NEAREST
        )

        self.gl.glMatrixMode(GL.GL_PROJECTION)
        self.gl.glLoadIdentity()
        self.gl.glOrtho(0, self.width(), self.height(), 0, -1, 1)

        self.gl.glMatrixMode(GL.GL_MODELVIEW)
        self.gl.glLoadIdentity()

        scaled_width = self.image_width * self.zoom
        scaled_height = self.image_height * self.zoom

        x_offset = (self.width() - scaled_width) / 2 + self.drag_offset[0]
        y_offset = (self.height() - scaled_height) / 2 + self.drag_offset[1]

        self.gl.glTranslatef(x_offset, y_offset, 0.0)
        self.gl.glScalef(self.zoom, self.zoom, 1.0)

        self.gl.glBegin(GL.GL_QUADS)
        self.gl.glTexCoord2f(0.0, 1.0)
        self.gl.glVertex2f(0.0, 0.0)
        self.gl.glTexCoord2f(1.0, 1.0)
        self.gl.glVertex2f(self.image_width, 0.0)
        self.gl.glTexCoord2f(1.0, 0.0)
        self.gl.glVertex2f(self.image_width, self.image_height)
        self.gl.glTexCoord2f(0.0, 0.0)
        self.gl.glVertex2f(0.0, self.image_height)
        self.gl.glEnd()

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