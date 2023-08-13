import sys
import os

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import OpenEXR
import Imath

from viewer import ImageViewerGLWidget
from utils.path_utils import PathSequence

def read_exr(image_path: str) -> np.ndarray:
    """Read an EXR image from file and return it as a NumPy array.

    Args:
        image_path (str): The path to the EXR image file.

    Returns:
        np.ndarray: The image data as a NumPy array.

    """
    # Open the EXR file for reading
    exr_file = OpenEXR.InputFile(image_path)

    # Get the image header
    header = exr_file.header()

    # Get the data window (bounding box) of the image
    data_window = header['dataWindow']

    # Get the channels present in the image
    channels = header['channels']

    # Calculate the width and height of the image
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1

    # Determine the channel keys
    channel_keys = 'RGB' if len(channels.keys()) == 3 else channels.keys()

    # Read all channels at once
    channel_data = exr_file.channels(channel_keys, Imath.PixelType(Imath.PixelType.FLOAT))

    # Create an empty NumPy array to store the image data
    image_data = np.zeros((height, width, len(channel_keys)), dtype=np.float32)

    # Populate the image data array
    for i, data in enumerate(channel_data):
        # Retrieve the pixel values for the channel
        pixels = np.frombuffer(data, dtype=np.float32)
        # Reshape the pixel values to match the image dimensions and store them in the image data array
        image_data[:, :, i] = pixels.reshape((height, width))

    return image_data

class Player(QtWidgets.QWidget):

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)

        self.path_sequence = PathSequence(image_path)

        self.first_frame, self.last_frame = self.path_sequence.get_frame_range()
        self.frame_count = self.path_sequence.get_frame_count_from_range()
        self.frame = self.first_frame
        # self.num_padding = path_sequence.padding
        
        image_path = self.path_sequence.get_frame_path(self.frame)
        

        self.image_data = self.read_image(image_path)
        self.image_dict = dict()
        self.image_dict[image_path] = self.image_data

        self.setup_ui()
        self.setup_signal()

    def setup_ui(self):
        self.play_forward_timer = QtCore.QTimer(self)
        self.play_forward_timer.setSingleShot(False)
        
        self.viewer = ImageViewerGLWidget(self, image_data=self.image_data)        

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addWidget(self.viewer)

        self.play_button = QtWidgets.QPushButton("play", self)
        self.stop_button = QtWidgets.QPushButton("stop", self)

        self.main_layout.addWidget(self.play_button)
        self.main_layout.addWidget(self.stop_button)

        self.setLayout(self.main_layout)

    def setup_signal(self):
        
        self.play_forward_timer.timeout.connect(self.play_forward)

        self.play_button.clicked.connect(self.play_forward_timer.start)
        self.stop_button.clicked.connect(self.play_forward_timer.stop)
        
        # self.play_forward_timer.start(8)

    def read_image(self, file_path: str()):
        image_data = read_exr(file_path)

        return image_data

    def play_forward(self):
        if self.frame == self.last_frame:
            self.frame = self.first_frame
        
        self.frame += 1

        image_path = self.path_sequence.get_frame_path(self.frame)
        
        if image_path in self.image_dict:
            
            image_data = self.image_dict[image_path]
        else:
            t0 = time.time()
            image_data = self.read_image(image_path)
            t1 = time.time()
            # print(t1-t0)

            self.image_dict[image_path] = image_data

        self.viewer.set_image(image_data)
        
if __name__ == "__main__":
    import time
    app = QtWidgets.QApplication(sys.argv)
    image_path = 'example_exr_plates\C0653.####.exr'
    
    win = Player(image_path)
    win.show()
    sys.exit( app.exec() )
