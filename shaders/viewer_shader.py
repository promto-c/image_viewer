from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets

import numpy as np

SHADERS_DIRECTORY = Path(__file__).parent
VIEWER_VERTEX_SHADER_PATH = SHADERS_DIRECTORY / 'viewer_vertex.glsl'
VIEWER_FRAGMENT_SHADER_PATH = SHADERS_DIRECTORY / 'viewer_fragment.glsl'

class ViewerShaderProgram(QtGui.QOpenGLShaderProgram):

    def __init__(self, parent: QtWidgets.QOpenGLWidget):
        super().__init__(parent.context())

        # Add shaders from files
        self.addShaderFromSourceFile(QtGui.QOpenGLShader.ShaderTypeBit.Vertex, VIEWER_VERTEX_SHADER_PATH.as_posix())
        self.addShaderFromSourceFile(QtGui.QOpenGLShader.ShaderTypeBit.Fragment, VIEWER_FRAGMENT_SHADER_PATH.as_posix())
        self.link()

        # Set uniform locations
        self.uniform_loc_transformation = self.uniformLocation("transformation")
        
        # Set uniform locations for image adjustments
        self.uniform_loc_lift = self.uniformLocation("lift")
        self.uniform_loc_gamma = self.uniformLocation("gamma")
        self.uniform_loc_gain = self.uniformLocation("gain")
        self.uniform_loc_saturation = self.uniformLocation("saturation")

        self.uniform_loc_is_texture = self.uniformLocation("isTexture")
        self.uniform_loc_custom_color = self.uniformLocation("color")

    def __enter__(self):
        self.bind()
        return self

    def __exit__(self, typ, val, tb):
        self.release()
    
    def id(self):
        return self.programId()

    def set_lift(self, lift: float):
        self.setUniformValue(self.uniform_loc_lift, lift)

    def set_gamma(self, gamma: float):
        self.setUniformValue(self.uniform_loc_gamma, gamma)

    def set_gain(self, gain: float):
        self.setUniformValue(self.uniform_loc_gain, gain)

    def set_saturation(self, saturation: float):
        self.setUniformValue(self.uniform_loc_saturation, saturation)

    def set_tranformation(self, tranformation: np.ndarray):
        self.setUniformValue(self.uniform_loc_transformation, QtGui.QMatrix4x4(tranformation.flatten()))

    def set_is_texture(self, is_texture: bool):
        self.setUniformValue(self.uniform_loc_is_texture, is_texture)

    def set_color(self, r, g, b, a):
        self.set_is_texture(False)
        self.setUniformValue(self.uniform_loc_custom_color, QtGui.QVector4D(r, g, b, a))

    def use_texture(self):
        self.set_is_texture(True)
