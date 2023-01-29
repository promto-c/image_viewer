from PyQt5 import QtGui, QtWidgets, QtOpenGL
from OpenGL.GL import *

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.image = None
        self.last_pos = None

    def loadImage(self, image_path):
        # code to load image from file and set it as self.image

    def paintGL(self):
        # code to render the image using OpenGL

    def mousePressEvent(self, event):
        # code to handle mouse press events for panning

    def mouseMoveEvent(self, event):
        # code to handle mouse move events for panning

    def wheelEvent(self, event):
        # code to handle mouse wheel events for zooming

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.gl_widget = GLWidget(self)
        self.setCentralWidget(self.gl_widget)

        # code to create a button to load image and connect it to the loadImage method of the GLWidget

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
