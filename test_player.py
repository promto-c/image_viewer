import sys
import os
from typing import Any, Callable, Dict, List, Tuple
from functools import lru_cache

import numpy as np
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets, uic

from theme import theme

from viewer import ImageViewerGLWidget
from entity import TrackerEntity
from utils.image_utils import read_exr
from utils.path_utils import PathSequence, PROJECT_ROOT
from utils.conversion_utils import to_uint8_gray, fps_to_interval_msc

PLAYER_WIDGET_UI = PROJECT_ROOT / 'ui/test_player_widget.ui'


def detect_features_with_grid(algorithm: Any, src_image_data: np.ndarray, 
                              grid_size: Tuple[int, int] = (18, 16), 
                              max_keypoints_per_cell: int = 1) -> List[cv2.KeyPoint]:
    """Detect keypoints in each grid cell of the image.

    Args:
        image (np.ndarray): Input image.
        grid_size (tuple): Number of rows and columns to divide the image.
        max_keypoints_per_cell (int): Maximum number of keypoints per cell.

    Returns:
        list: List of cv2.KeyPoint objects.
    """
    src_image_data = to_uint8_gray(src_image_data)

    height, width = src_image_data.shape[:2]
    cell_height, cell_width = height // grid_size[0], width // grid_size[1]

    all_keypoints = list()

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Define the cell's boundaries
            y_start, y_end = i * cell_height, (i + 1) * cell_height
            x_start, x_end = j * cell_width, (j + 1) * cell_width

            # Extract the cell from the image
            cell = src_image_data[y_start:y_end, x_start:x_end]

            # Detect keypoints in the cell
            keypoints = algorithm.detect(cell, None)

            # Sort keypoints by their response and keep the top 'max_keypoints_per_cell' keypoints
            keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)[:max_keypoints_per_cell]

            # Adjust keypoint positions to the global image (because they are currently local to the cell)
            for keypoint in keypoints:
                keypoint.pt = (keypoint.pt[0] + x_start, keypoint.pt[1] + y_start)

            all_keypoints.extend(keypoints)

    return all_keypoints


class ImageLoaderWorker(QtCore.QRunnable):
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

class Tracker:
    feature_detectors = dict(
        SIFT = cv2.SIFT_create,
        ORB = cv2.ORB_create,
        FAST = cv2.FastFeatureDetector_create,
    )

    feature_detector_to_parameters = dict(
        SIFT = dict(
            # nfeatures=0, 
            # nOctaveLayers=3, 
            contrastThreshold=0.002, 
            edgeThreshold=100, 
            sigma=0.3,
        ),
        ORB = dict(
            nfeatures=500, 
            scaleFactor=1.2, 
            nlevels=8, 
            edgeThreshold=31, 
            firstLevel=0, 
            WTA_K=2, 
            patchSize=31, 
            fastThreshold=20
        )
    )

    def __init__(self, detector: str, parameters: Dict[str, Any] = None):
        FeatureDetector = Tracker.feature_detectors.get(detector)

        if parameters is None:
            parameters = Tracker.feature_detector_to_parameters.get(detector)

        # TODO: Add try, except
        self.feature_detector = FeatureDetector(**parameters)

    # TODO: Set default to use whole image
    def detect_features(self, src_image_data: np.ndarray,
                        grid_size: Tuple[int, int] = (18, 16),
                        max_keypoints_per_cell: int = 1) -> List[cv2.KeyPoint]:
        """Detect keypoints in each grid cell of the image.

        Args:
            image (np.ndarray): Input image.
            grid_size (tuple): Number of rows and columns to divide the image.
            max_keypoints_per_cell (int): Maximum number of keypoints per cell.

        Returns:
            list: List of cv2.KeyPoint objects.
        """
        src_image_data = to_uint8_gray(src_image_data)

        height, width = src_image_data.shape[:2]
        cell_height, cell_width = height // grid_size[0], width // grid_size[1]

        all_keypoints = list()

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Define the cell's boundaries
                y_start, y_end = i * cell_height, (i + 1) * cell_height
                x_start, x_end = j * cell_width, (j + 1) * cell_width

                # Extract the cell from the image
                cell = src_image_data[y_start:y_end, x_start:x_end]

                # Detect keypoints in the cell
                keypoints = self.feature_detector.detect(cell, None)

                # Sort keypoints by their response and keep the top 'max_keypoints_per_cell' keypoints
                keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)[:max_keypoints_per_cell]

                # Adjust keypoint positions to the global image (because they are currently local to the cell)
                for keypoint in keypoints:
                    keypoint.pt = (keypoint.pt[0] + x_start, keypoint.pt[1] + y_start)

                all_keypoints.extend(keypoints)

        return all_keypoints
    
    @staticmethod
    def convert_keypoints(keypoints: List[Tuple[float, float]]) -> np.ndarray:
        keypoints = np.array(keypoints, dtype=np.float32)
        return keypoints.reshape(-1, 1, 2)

    def matching(self, src_image_data: np.ndarray, dst_image_data: np.ndarray, src_keypoints: List[Tuple[float, float]], 
                 dst_keypoints: List[Tuple[float, float]]) -> np.ndarray:
        src_image_data = to_uint8_gray(src_image_data)
        dst_image_data = to_uint8_gray(dst_image_data)

        src_keypoints = self.convert_keypoints(src_keypoints)
        dst_keypoints = self.convert_keypoints(dst_keypoints) if dst_keypoints is not None else None

        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Calculate optical flow
        dst_keypoints, status, err = cv2.calcOpticalFlowPyrLK(src_image_data, dst_image_data, src_keypoints, dst_keypoints, **lk_params)

        # Select good points (if status is 1)
        good_new = dst_keypoints[status == 1]
        # good_old = src_keypoints[status == 1]

        return good_new

class PlayerWidget(QtWidgets.QWidget):

    UI = PLAYER_WIDGET_UI

    entity_tree: QtWidgets.QTreeWidget

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

    # Tracker
    # -------
    track_previous_frame_button: QtWidgets.QPushButton
    track_backward_button: QtWidgets.QPushButton
    stop_track_button: QtWidgets.QPushButton
    track_forward_button: QtWidgets.QPushButton
    track_next_frame_button: QtWidgets.QPushButton

    feature_detector_combo_box: QtWidgets.QComboBox
    
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

        image_data = self.get_image_data(self.current_frame)
        self.viewer = ImageViewerGLWidget(self, image_data=image_data)

        self.thread_pool = QtCore.QThreadPool()

        num_cores = os.cpu_count()
        if num_cores is not None and num_cores > 1:
            self.thread_pool.setMaxThreadCount(num_cores - 1)
        else:
            self.thread_pool.setMaxThreadCount(1)

        self.play_forward_timer = QtCore.QTimer(self)
        self.play_backward_timer = QtCore.QTimer(self)

        self.tracker_entity = TrackerEntity([])

        self.viewer.vector_entities.append(self.tracker_entity)

        self.tracker = None

    def _setup_ui(self):

        self.set_playback_speed()
        
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

        self.feature_detector_combo_box.addItems(Tracker.feature_detectors)

        # Tracker
        # -------
        self.track_forward_timer = QtCore.QTimer(self)
        self.track_forward_timer.timeout.connect(self.track_next_frame)

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

        self.track_forward_button.clicked.connect(self.track_forward)
        self.track_next_frame_button.clicked.connect(self.track_next_frame)
        self.stop_track_button.clicked.connect(self.stop_track)

        # self.viewer.left_mouse_pressed.connect(self.add_tracker)
        # self.viewer.left_mouse_pressed.connect(self.test_sift)

        self.key_bind('j', self.play_backward)
        self.key_bind('k', self.stop_playback)
        self.key_bind('l', self.play_forward)

        self.key_bind('z', self.previous_frame)
        self.key_bind('x', self.next_frame)

    def track_forward(self):
        self.track_forward_timer.start()

    def track_next_frame(self):
        # Load the image
        image_data = self.get_image_data(self.current_frame)
        
        if self.current_frame not in self.tracker_entity.frame_to_points:

            feature_detector_name = self.feature_detector_combo_box.currentText()
            self.tracker = Tracker(detector=feature_detector_name)

            keypoint_cv_list = self.tracker.detect_features(src_image_data=image_data)

            # Convert keypoints to a list of (x, y) coordinates
            keypoints = [keypoint_cv.pt for keypoint_cv in keypoint_cv_list]

            for keypoint in keypoints:
                keypoint_gl = self.viewer.canvas_to_gl_coords(keypoint)

                # Add the point to the tracker entity
                self.tracker_entity.add_track_point(self.current_frame, keypoint_gl)
        
        else:
            keypoints_gl = self.tracker_entity.frame_to_points.get(self.current_frame)

            keypoints = [self.viewer.gl_to_canvas_coords(keypoint_gl) for keypoint_gl in keypoints_gl]

            test_kp_gl = keypoints_gl[0]

            test_kp = self.viewer.gl_to_canvas_coords(test_kp_gl)
            test_kp_g2 = self.viewer.canvas_to_gl_coords(test_kp)

        # NOTE:
        dst_image_data = self.get_image_data(self.current_frame+1)

        # Get dst_points, if next frame already has points
        if self.current_frame+1 in self.tracker_entity.frame_to_points:
             keypoints_gl = self.tracker_entity.frame_to_points.get(self.current_frame)
             dst_keypoints = [self.viewer.gl_to_canvas_coords(keypoint_gl) for keypoint_gl in keypoints_gl]

        else:
            dst_keypoints = None

        dst_keypoints = self.tracker.matching(image_data, dst_image_data, keypoints, dst_keypoints)

        for keypoint in dst_keypoints:
            keypoint_gl = self.viewer.canvas_to_gl_coords(keypoint)

            # Add the point to the tracker entity
            self.tracker_entity.add_track_point(self.current_frame+1, keypoint_gl)

        self.next_frame()

    def stop_track(self):
        self.track_forward_timer.stop()

    def add_tracker(self, event):
        # Convert the click position to the coordinate system you're using
        x, y = self.viewer.pixel_to_gl_coords(event.pos())

        # Add the point to the tracker entity
        self.tracker_entity.add_track_point((x, y))

        # Redraw the viewer to reflect the new track point
        self.viewer.update()

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
            worker = ImageLoaderWorker(self, frame, image_path)
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

        # TODO:
        self.viewer.set_image(image_data)
        self.viewer.set_frame(self.current_frame)

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
