# Standard Library Imports
# ------------------------
import os
from typing import Callable, Union

# Third Party Imports
# -------------------
from qtpy import QtCore, QtGui, QtWidgets, uic
from tablerqicon import TablerQIcon
from blackboard.utils.image_utils import ImageSequence
from blackboard.utils.key_binder import KeyBinder
from blackboard.widgets.double_spin_box import DoubleSpinBoxWidget

# Local Imports
# -------------
from viewer import ImageViewerGLWidget

from utils.path_utils import PACKAGE_ROOT
from utils.conversion_utils import fps_to_interval_msc
from nodes.node import Node


# Constant Definitions
# --------------------
PLAYER_WIDGET_UI = PACKAGE_ROOT / 'ui/player_widget.ui'


# Class Definitions
# -----------------
class ImageLoader(QtCore.QRunnable):
    def __init__(self, widget: 'PlayerWidget', frame: int):
        super().__init__()
        self.widget = widget
        self.frame = frame

    def run(self):
        image_data = self.widget.image.get_image_data(self.frame)
        try:
            self.widget.image_loaded_signal.emit(self.frame, image_data)
        except RuntimeError as e:
            pass


class PlayerWidget(QtWidgets.QWidget):

    UI = PLAYER_WIDGET_UI
    TITLE = 'Player'

    prefetch_button: QtWidgets.QPushButton
    prefetch_progress_bar: QtWidgets.QProgressBar
    num_cores_spin_box: QtWidgets.QSpinBox
    cpu_button: QtWidgets.QPushButton

    gain_button: QtWidgets.QPushButton
    gamma_button: QtWidgets.QPushButton
    lift_button: QtWidgets.QPushButton

    top_right_layout: QtWidgets.QHBoxLayout
    center_layout: QtWidgets.QVBoxLayout

    playback_speed_button: QtWidgets.QPushButton
    playback_speed_combo_box: QtWidgets.QComboBox

    current_frame_spin_box: QtWidgets.QSpinBox
    play_backward_button: QtWidgets.QPushButton
    stop_button: QtWidgets.QPushButton
    play_forward_button: QtWidgets.QPushButton

    start_frame_button: QtWidgets.QPushButton
    end_frame_button: QtWidgets.QPushButton

    start_frame_spin_box: QtWidgets.QSpinBox
    frame_slider: QtWidgets.QSlider
    end_frame_spin_box: QtWidgets.QSpinBox

    image_loaded_signal = QtCore.Signal(int, object)
    
    def __init__(self, input_path: str = str(), parent=None):
        super().__init__(parent)
        uic.loadUi(str(self.UI), self)

        if input_path:
            self.image = ImageSequence(input_path)
        else:
            self.image = None

        # Initialize setup
        self.__init_attributes()
        self.__init_ui()
        self.__init_signal_connections()

    def __init_attributes(self):
        """Initialize the attributes.
        """
        self.current_frame = 0.0

        self.viewer = ImageViewerGLWidget(parent=self)

        self.thread_pool = self._create_thread_pool()

        self.play_forward_timer = QtCore.QTimer(self)
        self.play_backward_timer = QtCore.QTimer(self)

        self.tabler_icon = TablerQIcon(opacity=0.8)

        # Widget
        # ------
        self.gain_spin_box_widget = DoubleSpinBoxWidget(default_value=1.0, parent=self)
        self.gain_spin_box_widget.setToolTip('Gain')
        self.gamma_spin_box_widget = DoubleSpinBoxWidget(default_value=1.0, parent=self)
        self.gamma_spin_box_widget.setToolTip('Gamma')

    def __init_ui(self):
        """Initialize the UI of the widget.
        """
        # Setup Main Widget
        # -----------------
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(TablerQIcon.player_play)

        # Top Right
        # ---------
        self.top_right_layout.addWidget(self.gain_spin_box_widget)
        self.top_right_layout.addWidget(self.gamma_spin_box_widget)

        # Set the focus policy to accept focus, and set the initial focus
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        # Viewer
        # ------
        self.set_image(self.image)
        self.center_layout.addWidget(self.viewer)

        # 
        self.set_playback_speed()

        self.prefetch_progress_bar.setMaximum(0)
        self.prefetch_progress_bar.setMinimum(0)
        self.prefetch_progress_bar.setValue(0)

        # Set Icons
        # ---------
        self.cpu_button.setIcon(self.tabler_icon.cpu_2)

        self.gain_spin_box_widget.setIcon(self.tabler_icon.flip.brightness_half)
        self.gamma_spin_box_widget.setIcon(self.tabler_icon.contrast)

        self.play_backward_button.setIcon(self.tabler_icon.flip.player_play)
        self.stop_button.setIcon(self.tabler_icon.player_stop)
        self.play_forward_button.setIcon(self.tabler_icon.player_play)

        self.playback_speed_button.setIcon(self.tabler_icon.keyframes)

        self.start_frame_button.setIcon(self.tabler_icon.brackets_contain_start)
        self.end_frame_button.setIcon(self.tabler_icon.brackets_contain_end)

    def __init_signal_connections(self):
        """Initialize signal-slot connections.
        """
        self.prefetch_button.clicked.connect(self.prefetch)

        self.playback_speed_combo_box.currentTextChanged.connect(self.set_playback_speed)
        
        self.play_forward_timer.timeout.connect(self.next_frame)
        self.play_backward_timer.timeout.connect(self.previous_frame)

        self.play_forward_button.clicked.connect(self.play_forward)
        self.play_backward_button.clicked.connect(self.play_backward)
        self.stop_button.clicked.connect(self.stop_playback)

        # self.lift_spin_box.valueChanged.connect(self.set_lift)
        self.gamma_spin_box_widget.valueChanged.connect(self.viewer.set_gamma)
        self.gain_spin_box_widget.valueChanged.connect(self.viewer.set_gain)

        self.current_frame_spin_box.valueChanged.connect(self.set_frame)
        self.frame_slider.valueChanged.connect(self.set_frame)

        self.image_loaded_signal.connect(self.update_progress_bar)

        # Key Binds
        # ---------
        KeyBinder.bind_key('j', parent_widget=self, callback=self.play_backward)
        KeyBinder.bind_key('k', parent_widget=self, callback=self.stop_playback)
        KeyBinder.bind_key('l', parent_widget=self, callback=self.play_forward)

        KeyBinder.bind_key('z', parent_widget=self, callback=self.previous_frame)
        KeyBinder.bind_key('x', parent_widget=self, callback=self.next_frame)

    def _create_thread_pool(self):
        thread_pool = QtCore.QThreadPool()

        num_cores = os.cpu_count()
        if num_cores is not None and num_cores > 1:
            thread_pool.setMaxThreadCount(num_cores - 1)
        else:
            thread_pool.setMaxThreadCount(1)

        return thread_pool

    def set_image(self, image: Union[ImageSequence, Node] = None):
        self.image = image

        if isinstance(image, (ImageSequence, Node)) and self.image.frame_range():
            self.first_frame, self.last_frame = self.image.frame_range()
        else:
            self.first_frame, self.last_frame = 0, 0

        # self.current_frame = self.first_frame

        self.viewer.set_image(self.image)

        if isinstance(self.image, Node):
            self.image.set_player(self)
            # self.viewer.vector_entities.extend(self.image.vector_entities)
        self.viewer.fit_in_view()

        self.frame_slider.setMinimum(self.first_frame)
        self.frame_slider.setMaximum(self.last_frame)

        self.start_frame_spin_box.setValue(self.first_frame)
        self.end_frame_spin_box.setValue(self.last_frame)

    @QtCore.Slot()
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

        for frame in range(int(start_frame), int(end_frame) + 1):
            # Create a worker and pass the read_image function to it
            worker = ImageLoader(self, frame)
            # Start the worker thread
            self.thread_pool.start(worker)
            # self.thread_pool.waitForDone(0)

    def set_playback_speed(self, playback_fps: float = 24.0):
        try:
            playback_fps = float(playback_fps)
        except ValueError as e:
            return

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

        self.current_frame_spin_box.setValue(int(self.current_frame))
        self.frame_slider.setValue(self.current_frame)

        self.viewer.set_frame(self.current_frame)

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
    import blackboard as bb
    import sys
    
    app = QtWidgets.QApplication(sys.argv)
    bb.theme.set_theme(app, theme='dark')

    image_path = 'example_exr_plates\C0653.####.exr'
    player_widget = PlayerWidget()
    player_widget.show()

    # Test chnage image
    image_sequence = ImageSequence(image_path)
    player_widget.set_image(image_sequence)

    sys.exit(app.exec())
