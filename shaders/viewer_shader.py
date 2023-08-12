from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np

SHADERS_DIRECTORY = Path(__file__).parent
VIEWER_VERTEX_SHADER_PATH = SHADERS_DIRECTORY / 'viewer_vertex.glsl'
VIEWER_FRAGMENT_SHADER_PATH = SHADERS_DIRECTORY / 'viewer_fragment.glsl'

class ViewerShaderProgram(QtGui.QOpenGLShaderProgram):

    def __init__(self, parent: QtWidgets.QOpenGLWidget):
        super().__init__(parent.context())

        # self.shader_program = QtGui.QOpenGLShaderProgram(self.context())
        self.addShaderFromSourceFile(QtGui.QOpenGLShader.ShaderTypeBit.Vertex, str(VIEWER_VERTEX_SHADER_PATH))
        self.addShaderFromSourceFile(QtGui.QOpenGLShader.ShaderTypeBit.Fragment, str(VIEWER_FRAGMENT_SHADER_PATH))
        self.link()

        # Set uniform locations
        self.uniform_loc_transformation = self.uniformLocation("transformation")
        # Set uniform locations, image adjustments
        self.uniform_loc_lift = self.uniformLocation("lift")
        self.uniform_loc_gamma = self.uniformLocation("gamma")
        self.uniform_loc_gain = self.uniformLocation("gain")

    def __enter__(self):
        self.bind()
        return self

    def __exit__( self, typ, val, tb ):
        self.release()

    def set_lift(self, lift: float):
        self.setUniformValue(self.uniform_loc_lift, lift)

    def set_gamma(self, gamma: float):
        self.setUniformValue(self.uniform_loc_gamma, gamma)

    def set_gain(self, gain: float):
        self.setUniformValue(self.uniform_loc_gain, gain)

    def set_tranformation(self, tranformation: np.ndarray):
        self.setUniformValue(self.uniform_loc_transformation, QtGui.QMatrix4x4(tranformation.flatten()))
