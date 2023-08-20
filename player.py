import sys
import os
from typing import Callable
from functools import lru_cache

from PyQt5 import QtCore, QtGui, QtWidgets, uic

from theme import theme

from viewer import ImageViewerGLWidget
from utils.image_utils import read_exr
from utils.path_utils import PathSequence, PROJECT_ROOT

PLAYER_WIDGET_UI = PROJECT_ROOT / 'ui/player_widget.ui'

def fps_to_interval_msc(fps) -> int:
    """Converts frames per second to time interval in milliseconds.

    Args:
        fps (float): Frames per second.

    Returns:
        float: Time interval between frames in milliseconds.

    Raises:
        ValueError: If FPS is not a positive value.
    """
    if fps <= 0:
        raise ValueError("FPS must be a positive value.")
    
    return int(1000 / fps)

class ImageLoader(QtCore.QRunnable):
    def __init__(self, widget, frame, file_path):
        super().__init__()
        self.widget = widget
        self.frame = frame
        self.file_path = file_path

    def run(self):
        image_data = self.widget.read_image(self.file_path)
        try:
            self.widget.image_loaded_signal.emit(self.frame, image_data)
        except RuntimeError as e:
            pass

class PlayerWidget(QtWidgets.QWidget):

    UI = PLAYER_WIDGET_UI

    prefetch_button: QtWidgets.QPushButton
    prefetch_progress_bar: QtWidgets.QProgressBar

    lift_spin_box: QtWidgets.QDoubleSpinBox
    gamma_spin_box: QtWidgets.QDoubleSpinBox
    gain_spin_box: QtWidgets.QDoubleSpinBox

    center_layout: QtWidgets.QVBoxLayout

    playback_speed_combo_box: QtWidgets.QComboBox

    current_frame_spin_box: QtWidgets.QSpinBox
    play_backward_button: QtWidgets.QPushButton
    stop_button: QtWidgets.QPushButton
    play_forward_button: QtWidgets.QPushButton

    start_frame_spin_box: QtWidgets.QSpinBox
    frame_slider: QtWidgets.QSlider
    end_frame_spin_box: QtWidgets.QSpinBox

    image_loaded_signal = QtCore.pyqtSignal(int, object)
    
    def __init__(self, input_path: str, parent=None):
        super().__init__(parent)
        uic.loadUi(self.UI, self)

        self.input_path = input_path

        # Set up the initial values
        self._setup_attributes()
        # Set up the UI
        self._setup_ui()
        # Set up signal connections
        self._setup_signal_connections()

    def _setup_attributes(self):
        self.path_sequence = PathSequence(self.input_path)
        self.first_frame, self.last_frame = self.path_sequence.get_frame_range()

        self.current_frame = self.first_frame

        self.thread_pool = QtCore.QThreadPool()

        num_cores = os.cpu_count()
        if num_cores is not None and num_cores > 1:
            self.thread_pool.setMaxThreadCount(num_cores - 1)
        else:
            self.thread_pool.setMaxThreadCount(1)

        self.play_forward_timer = QtCore.QTimer(self)
        self.play_backward_timer = QtCore.QTimer(self)

    def _setup_ui(self):

        self.set_playback_speed()

        image_data = self.get_image_data(self.current_frame)
        self.viewer = ImageViewerGLWidget(self, image_data=image_data)

        self.frame_slider.setMaximum(self.last_frame)

        self.start_frame_spin_box.setValue(self.first_frame)
        self.end_frame_spin_box.setValue(self.last_frame)

        self.center_layout.addWidget(self.viewer)

        # Set the focus policy to accept focus, and set the initial focus
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        self.prefetch_progress_bar.setMaximum(0)
        self.prefetch_progress_bar.setMinimum(0)
        self.prefetch_progress_bar.setValue(0)

    def _setup_signal_connections(self):

        self.prefetch_button.clicked.connect(self.prefetch)

        self.playback_speed_combo_box.currentTextChanged.connect(self.set_playback_speed)
        
        self.play_forward_timer.timeout.connect(self.next_frame)
        self.play_backward_timer.timeout.connect(self.previous_frame)

        self.play_forward_button.clicked.connect(self.play_forward)
        self.play_backward_button.clicked.connect(self.play_backward)
        self.stop_button.clicked.connect(self.stop_playback)

        self.lift_spin_box.valueChanged.connect(self.set_lift)
        self.gamma_spin_box.valueChanged.connect(self.set_gamma)
        self.gain_spin_box.valueChanged.connect(self.set_gain)

        self.current_frame_spin_box.valueChanged.connect(self.set_frame)
        self.frame_slider.valueChanged.connect(self.set_frame)

        self.image_loaded_signal.connect(self.update_progress_bar)

        self.key_bind('j', self.play_backward)
        self.key_bind('k', self.stop_playback)
        self.key_bind('l', self.play_forward)

    @QtCore.pyqtSlot()
    def update_progress_bar(self):
        current_value = self.prefetch_progress_bar.value()
        self.prefetch_progress_bar.setValue(current_value + 1)

    def prefetch(self):
        # Calculate the range of frames to prefetch
        start_frame = self.current_frame
        end_frame = self.end_frame_spin_box.value()

        # Set the range for the progress bar
        total_frames = end_frame - start_frame + 1
        self.prefetch_progress_bar.setMaximum(total_frames)
        self.prefetch_progress_bar.setValue(0)

        for frame in range(start_frame, end_frame + 1):
            image_path = self.path_sequence.get_frame_path(frame)
            # Create a worker and pass the read_image function to it
            worker = ImageLoader(self, frame, image_path)
            # Start the worker thread
            self.thread_pool.start(worker)
            # self.thread_pool.waitForDone(0)

    def key_bind(self, key_sequence: str, function: Callable):
        # Create a shortcut
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key_sequence), self)
        # Connect the activated signal of the shortcut to the slot
        shortcut.activated.connect(function)

    def set_playback_speed(self, playback_fps: float = 48.0):
        playback_fps = float(playback_fps)

        interval = fps_to_interval_msc(playback_fps)

        self.play_forward_timer.setInterval(interval)
        self.play_backward_timer.setInterval(interval)
        
    def stop_playback(self):
        self.play_forward_timer.stop()
        self.play_backward_timer.stop()

    def play_forward(self):
        self.stop_playback()
        self.play_forward_timer.start()

    def play_backward(self):
        self.stop_playback()
        self.play_backward_timer.start()

    def set_frame(self, frame_number: int):
        self.current_frame = frame_number
        self.current_frame_spin_box.setValue(self.current_frame)
        self.frame_slider.setValue(self.current_frame)

        image_data = self.get_image_data(self.current_frame)
        self.viewer.set_image(image_data)

    def get_image_data(self, frame: int):
        image_path = self.path_sequence.get_frame_path(frame)
        return self.read_image(image_path)

    def set_lift(self, lift_value: float):
        self.viewer.lift = lift_value
        self.viewer.update()

    def set_gamma(self, gamma_value: float):
        self.viewer.gamma = gamma_value
        self.viewer.update()

    def set_gain(self, gain_value: float):
        self.viewer.gain = gain_value
        self.viewer.update()

    @lru_cache(maxsize=400)
    def read_image(self, file_path: str()):
        image_data = read_exr(file_path)

        return image_data

    def next_frame(self, increment: int = 1):
        if self.current_frame == self.last_frame:
            next_frame = self.first_frame
        else:
            next_frame = self.current_frame + increment

        self.set_frame(next_frame)

    def previous_frame(self, increment: int = 1):
        if self.current_frame == self.first_frame:
            next_frame = self.last_frame
        else:
            next_frame = self.current_frame - increment

        self.set_frame(next_frame)

    def closeEvent(self, event):
        self.thread_pool.clear()  # clears the queued tasks, but won't stop currently running threads
        super().closeEvent(event)

if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    theme.set_theme(app, theme='dark')

    image_path = 'example_exr_plates\C0653.####.exr'
    
    player_widget = PlayerWidget(image_path)
    player_widget.show()
    sys.exit(app.exec())
