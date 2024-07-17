
# Standard Library Imports
# ------------------------
from typing import Any, Callable, Dict, List, Tuple, Union, TYPE_CHECKING
from pathlib import Path
import sys
import numpy as np
import cv2
from enum import Enum

# Third Party Imports
# -------------------
from qtpy import QtCore, QtGui, QtWidgets, uic
from tablerqicon import TablerQIcon
from NodeGraphQt import BaseNode

# Local Imports
# -------------
from entity import TrackPointsEntity
from nodes.node import Node
from utils.conversion_utils import to_uint8_gray

if TYPE_CHECKING:
    from player import PlayerWidget

TRACKER_NODE_ROOT = Path(__file__).parent

# Path to UI file
TRACKER_NODE_PANEL_UI = TRACKER_NODE_ROOT / 'ui/tracker_panel.ui'

class TrackerNodePanel(QtWidgets.QWidget):
    """A PyQt5 widget with a user interface created from a .ui file.
    
    Attributes:
        ...
    """
    feature_detector_combo_box: QtWidgets.QComboBox
    detect_feature_button: QtWidgets.QToolButton

    track_previous_frame_button: QtWidgets.QPushButton
    track_backward_button: QtWidgets.QPushButton
    stop_track_button: QtWidgets.QPushButton
    track_forward_button: QtWidgets.QPushButton
    track_next_frame_button: QtWidgets.QPushButton

    # Initialization and Setup
    # ------------------------
    def __init__(self, parent=None):
        """Initialize the widget and set up the UI, signal connections, and icon.

        Args:
            ...
        """
        # Initialize the super class
        super().__init__(parent)

        # Load the .ui file using the uic module
        uic.loadUi(str(TRACKER_NODE_PANEL_UI), self)

        # Set up the initial attributes
        self.__init_attributes()
        # Set up the UI
        self.__init_ui()
        # Set up signal connections
        self.__init_signal_connections()
        # Set up the icons
        self._setup_icons()

    def __init_attributes(self):
        """Set up the initial values for the widget.
        """
        # Attributes
        # ------------------
        self.tabler_qicon =  TablerQIcon()

        # Private Attributes
        # ------------------

    def __init_ui(self):
        """Set up the UI for the widget, including creating widgets and layouts.
        """
        # Create widgets and layouts here
        pass

    def __init_signal_connections(self):
        """Set up signal connections between widgets and slots.
        """
        # Connect signals to slots here
        pass

    def _setup_icons(self):
        """Set the icons for the widgets.
        """
        self.detect_feature_button.setIcon(self.tabler_qicon.photo_search)

        self.track_previous_frame_button.setIcon(self.tabler_qicon.arrow_bar_to_left)
        self.track_backward_button.setIcon(self.tabler_qicon.arrow_left)
        self.stop_track_button.setIcon(self.tabler_qicon.player_stop)
        self.track_forward_button.setIcon(self.tabler_qicon.arrow_right)
        self.track_next_frame_button.setIcon(self.tabler_qicon.arrow_bar_to_right)

    # Private Methods
    # ---------------

    # Extended Methods
    # ----------------

    # Special Methods
    # ---------------

    # Event Handling or Override Methods
    # ----------------------------------
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """Handle key press events.
        """
        # Handle key press events here
        super().keyPressEvent(event)

class FeatureDetector(Enum):
    SIFT = "SIFT"
    ORB = "ORB"
    FAST = "FAST"

    @staticmethod
    def _get_detector_mapping():
        return {
            FeatureDetector.SIFT: cv2.SIFT_create,
            FeatureDetector.ORB: cv2.ORB_create,
            FeatureDetector.FAST: cv2.FastFeatureDetector_create,
        }

    @staticmethod
    def _get_default_parameters_mapping():
        return {
            FeatureDetector.SIFT: dict(
                # nfeatures=0,
                # nOctaveLayers=3,
                contrastThreshold=0.002, 
                edgeThreshold=100, 
                sigma=0.3,
            ),
            FeatureDetector.ORB: dict(
                nfeatures=500, 
                scaleFactor=1.2, 
                nlevels=8, 
                edgeThreshold=31, 
                firstLevel=0, 
                WTA_K=2, 
                patchSize=31, 
                fastThreshold=20
            ),
            FeatureDetector.FAST: dict(),
        }

    @staticmethod
    def get_names():
        return [member.name for member in FeatureDetector]

    def __str__(self):
        return self.value

    def __call__(self, **params):
        params = params or self._get_default_parameters_mapping()[self]
        return self._get_detector_mapping()[self](**params)

class Tracker:

    def __init__(self, detector: str = None, parameters: Dict[str, Any] = None):

        if detector is not None:
            self.use_detector(detector, parameters)

    def use_detector(self, detector: Union[FeatureDetector, Any], parameters: Dict[str, Any] = {}):
        if isinstance(detector, FeatureDetector):
            # TODO: Add try, except
            self.feature_detector = detector(**parameters)
        else:
            self.feature_detector = detector

    # TODO: Implemnet this
    def use_matcher(self):
        print('Not implement')
        pass

    # TODO: Set default to use whole image
    def detect_features(self, src_image_data: np.ndarray,
                        grid_size: Tuple[int, int] = (9, 7),
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

                    all_keypoints.append(keypoint.pt)

        return all_keypoints

    @staticmethod
    def convert_keypoints(keypoints: List[Tuple[float, float]]) -> np.ndarray:
        keypoints = np.array(keypoints, dtype=np.float32)
        return keypoints.reshape(-1, 1, 2)

    def matching(self, src_image_data: np.ndarray, dst_image_data: np.ndarray, src_keypoints: List[Tuple[float, float]], 
                 dst_keypoints: List[Tuple[float, float]], win_size=(50, 50)) -> np.ndarray:
        src_image_data = to_uint8_gray(src_image_data)
        dst_image_data = to_uint8_gray(dst_image_data)

        src_keypoints = self.convert_keypoints(src_keypoints)
        dst_keypoints = self.convert_keypoints(dst_keypoints) if dst_keypoints is not None else None

        # Parameters for Lucas-Kanade optical flow
        lk_params = dict(winSize=win_size, maxLevel=8, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Build pyramids
        # src_pyramid = cv2.buildOpticalFlowPyramid(src_image_data, win_size, max_level)[1]
        # dst_pyramid = cv2.buildOpticalFlowPyramid(dst_image_data, win_size, max_level)[1]

        # Calculate optical flow
        dst_keypoints, status, err = cv2.calcOpticalFlowPyrLK(src_image_data, dst_image_data, src_keypoints, dst_keypoints, **lk_params)

        # Select good points (if status is 1)
        good_new = [dst_keypoints[i][0] if status[i] else None for i in range(len(dst_keypoints))]

        # good_new = dst_keypoints[status == 1]
        # good_old = src_keypoints[status == 1]

        return good_new
    
class TrackerNode(Node):

    # unique node identifier.
    __identifier__ = 'nodes.tracker'

    # initial default node name.
    NODE_NAME = 'Tracker'

    def __init__(self, player: 'PlayerWidget' = None) -> None:
        super().__init__()

        # TODO: Use image data from input node instead of player or viewer
        self.player = player

        # Initialize setup
        self.__init_attributes()
        self.__init_ports()
        self.__init_ui()
        self.__init_signal_connections()

    def __init_attributes(self):
        # Tools
        # -----
        self.tracker = Tracker()

        # Panels
        # ------
        self.panel = TrackerNodePanel()

        # Tracker
        # -------
        self.vector_entities = list()
        self.track_forward_timer = QtCore.QTimer(self.panel)
        self.track_backward_timer = QtCore.QTimer(self.panel)

        self.track_points_entity = TrackPointsEntity()
        self.vector_entities.append(self.track_points_entity)

    def __init_ports(self):
        # create input ports
        self.add_input('in 1')

    def __init_ui(self):
        # Panels
        # ------
        # self.panel.feature_detector_combo_box.addItems(list(FeatureDetector))
        for detector_enum in FeatureDetector:
             self.panel.feature_detector_combo_box.addItem(str(detector_enum), userData=detector_enum)

        # self.comboBox.addItem("Item 1", userData="Data 1")
        self.add_combo_menu('Feature Detector', '', FeatureDetector.get_names())

    def __init_signal_connections(self):
        # Tracker
        # -------
        self.track_forward_timer.timeout.connect(self.track_next_frame)
        self.track_backward_timer.timeout.connect(self.track_previous_frame)

        # Panel
        # -----
        self.panel.detect_feature_button.clicked.connect(self.detect_features)

        self.panel.track_previous_frame_button.clicked.connect(self.track_previous_frame)
        self.panel.track_backward_button.clicked.connect(self.track_backward)
        self.panel.track_forward_button.clicked.connect(self.track_forward)
        self.panel.track_next_frame_button.clicked.connect(self.track_next_frame)
        self.panel.stop_track_button.clicked.connect(self.stop_track)

    def get_input_image_data(self, frame):
        predecessor_node = self.get_input_node(0)

        if not predecessor_node:
            return None

        image_data = predecessor_node.get_image_data(frame)

        return image_data

    def get_image_data(self, frame):
        predecessor_node = self.get_input_node(0)

        if not predecessor_node:
            return None

        image_data = predecessor_node.get_image_data(frame)

        return image_data
    
    def set_player(self, player):
        self.player = player
        self.viewer = self.player.viewer
    # Tracker
    # -------
    def track_backward(self):
        self.track_backward_timer.start()

    def track_forward(self):
        self.track_forward_timer.start()

    def track_previous_frame(self):
        self.track(frame_increment=(-1))
        self.player.previous_frame()

    def track_next_frame(self):
        self.track(frame_increment=1)
        self.player.next_frame()

    def detect_features(self, src_frame):
        feature_detector_name = self.panel.feature_detector_combo_box.currentData()
        self.tracker.use_detector(feature_detector_name)
    
        # TODO: get image from input instead of player
        src_image_data = self.get_input_image_data(src_frame)
        keypoints = self.tracker.detect_features(src_image_data=src_image_data)
        # Add points to the tracker entity
        id_to_track_point = self.track_points_entity.add_track_points(src_frame, keypoints)

        if self.viewer is not None:
            self.viewer.update()

        return id_to_track_point

    def track(self, frame_increment: int):

        feature_detector_name = self.panel.feature_detector_combo_box.currentData()
        self.tracker.use_detector(feature_detector_name)

        # TODO: Check dst exist
        src_frame = self.player.current_frame
        dst_frame = src_frame + frame_increment

        # TODO: get image from input instead of player
        # Load the image
        src_image_data = self.get_input_image_data(src_frame)
        dst_image_data = self.get_input_image_data(dst_frame)

        # Get all track points
        id_to_track_point = self.track_points_entity.props.track_points.get_value()
        # Get all track points id in src frame
        track_point_ids = [track_point_id for track_point_id, track_point_property in id_to_track_point.items() if track_point_property[src_frame] is not None]

        if not track_point_ids:
            id_to_track_point = self.detect_features(src_frame)
            # # Add points to the tracker entity
            track_point_ids = id_to_track_point.keys()

        src_keypoints = [id_to_track_point[track_point_id][src_frame] for track_point_id in track_point_ids]
        dst_keypoints = [id_to_track_point[track_point_id][dst_frame] for track_point_id in track_point_ids]

        dst_keypoints = self.tracker.matching(src_image_data, dst_image_data, src_keypoints, dst_keypoints)

        # Contruct id to poisition dict
        update_id_to_position = {
            track_point_id: position for track_point_id, position in zip(track_point_ids, dst_keypoints) if position is not None
        }

        # Add points to the tracker entity
        self.track_points_entity.update_track_points(dst_frame, update_id_to_position)

    def stop_track(self):
        self.track_forward_timer.stop()
        self.track_backward_timer.stop()
        
    def add_tracker(self, event):
        if self.viewer is not None:
            # Convert the click position to the coordinate system you're using
            x, y = self.viewer.diaplay_to_gl_coords(event.pos())

            # Add the point to the tracker entity
            self.track_points_entity.add_track_point((x, y))

            # Redraw the viewer to reflect the new track point
            self.viewer.update()

# TODO: Some code to register node

def main():
    """Create the application and main window, and show the widget.
    """
    # Create the application and the main window
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()

    # Create an instance of the widget and set it as the central widget
    widget = TrackerNodePanel()
    window.setCentralWidget(widget)

    # Show the window and run the application
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
