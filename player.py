# Standard Library Imports
# ------------------------
import os, re
from typing import Any, Callable, Union, TYPE_CHECKING, List
if TYPE_CHECKING:
    import numpy as np


# Third Party Imports
# -------------------
from qtpy import QtCore, QtGui, QtWidgets, uic
from tablerqicon import TablerQIcon
from blackboard.utils.image_utils import ImageSequence
from blackboard.utils.key_binder import KeyBinder
from blackboard.widgets.double_spin_box import DoubleSpinBoxWidget
from blackboard.widgets.frame_indicator_widget import FrameIndicatorBar, FrameStatus
from blackboard.widgets.gallery_view import GalleryWidget
from blackboard.utils.file_path_utils import SequenceFileUtil

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
class GroupWidget(QtWidgets.QWidget):
    def __init__(self, *buttons):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)  # No space between buttons
        self.setMaximumHeight(22)

        # Set custom properties for styling based on button position
        if buttons:
            buttons[0].setProperty("position", "first")
            buttons[-1].setProperty("position", "last")

        for button in buttons:
            layout.addWidget(button)

        # Apply the stylesheet to the container (QFrame) only
        self.setStyleSheet("""
            QWidget {
                border-radius: 0;
                border-left: none;
                border-right: none;
            }
            QWidget[position="first"] {
                border-top-left-radius: 4;
                border-bottom-left-radius: 4;
                border-left: 1px solid gray;
            }
            QWidget[position="last"] {
                border-top-right-radius: 4;
                border-bottom-right-radius: 4;
                border-right: 1px solid gray;
            }
        """)

class ImageLoader(QtCore.QRunnable):
    def __init__(self, widget: 'PlayerWidget', frame: int):
        super().__init__()
        self.widget = widget
        self.frame = frame

    def run(self):
        self.widget.image_loading_signal.emit(self.frame)
        _image_data = self.widget.image.get_image_data(self.frame)
        try:
            self.widget.image_loaded_signal.emit(self.frame)
        except RuntimeError as e:
            pass

class ControllerBarWidget(QtWidgets.QToolBar):
    """A toolbar for controlling playback and frame settings in the PlayerWidget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabler_icon = TablerQIcon(opacity=0.8)
        self.setMovable(False)  # Keep toolbar fixed
        self.__init_ui()

    def __init_ui(self):
        """Initialize the UI components of the controller bar.
        """
        self.setStyleSheet('''
            QToolBar {
                background: transparent;
            }''')

        # Playback Controls
        self.play_backward_button = QtWidgets.QPushButton(self.tabler_icon.flip.player_play, '', self)
        self.stop_button = QtWidgets.QPushButton(self.tabler_icon.player_stop, '', self)
        self.play_forward_button = QtWidgets.QPushButton(self.tabler_icon.player_play, '', self)
        playback_widget = GroupWidget(self.play_backward_button, self.stop_button, self.play_forward_button)

        # Playback Speed Controls
        self.playback_speed_button = QtWidgets.QPushButton(self.tabler_icon.keyframes, '', self)
        self.playback_speed_button.setEnabled(False)
        self.playback_speed_combo_box = QtWidgets.QComboBox(self)
        self.playback_speed_combo_box.addItems(["24", "60"])
        self.playback_speed_combo_box.setEditable(True)
        self.playback_speed_combo_box.setMinimumSize(QtCore.QSize(48, 0))
        playback_speed_widget = GroupWidget(self.playback_speed_button, self.playback_speed_combo_box)

        # Frame Range Controls
        self.start_frame_button = QtWidgets.QPushButton(self.tabler_icon.brackets_contain_start, '', self)
        self.start_frame_button.setEnabled(False)
        self.start_frame_spin_box = QtWidgets.QSpinBox(self)
        self.start_frame_spin_box.setButtonSymbols(QtWidgets.QSpinBox.ButtonSymbols.NoButtons)

        self.end_frame_button = QtWidgets.QPushButton(self.tabler_icon.brackets_contain_end, '', self)
        self.end_frame_button.setEnabled(False)
        self.end_frame_spin_box = QtWidgets.QSpinBox(self)
        self.end_frame_spin_box.setMaximum(65535)
        self.end_frame_spin_box.setButtonSymbols(QtWidgets.QSpinBox.ButtonSymbols.NoButtons)

        # Wrap the indicator layout in a QWidget for the toolbar
        frame_indicator_widget = QtWidgets.QFrame(self)
        frame_indicator_widget.setStyleSheet('''
            QFrame {
                border: 2 solid #444;
                background-color: rgba(31, 31, 31, 0.6);
            }
        ''')
        # Frame Indicator Bar and Slider
        self.frame_indicator_bar = FrameIndicatorBar()
        self.frame_indicator_bar.setMaximumHeight(2)
        self.frame_slider = QtWidgets.QSlider(self)
        self.frame_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)

        # Frame indicator layout within toolbar
        frame_indicator_layout = QtWidgets.QVBoxLayout(frame_indicator_widget)
        frame_indicator_layout.setContentsMargins(0, 0, 0, 0)
        frame_indicator_layout.setSpacing(0)
        frame_indicator_layout.addWidget(self.frame_indicator_bar)
        frame_indicator_layout.addWidget(self.frame_slider)

        frame_range_widget = GroupWidget(
            self.start_frame_button, self.start_frame_spin_box, 
            frame_indicator_widget, 
            self.end_frame_spin_box, self.end_frame_button,
        )

        # Current Frame Control
        self.current_frame_spin_box = QtWidgets.QSpinBox(self)
        self.current_frame_spin_box.setButtonSymbols(QtWidgets.QSpinBox.NoButtons)
        self.current_frame_spin_box.setStyleSheet('''
            QSpinBox {
               border-radius: 0px;
            }
        ''')
        self.current_frame_spin_box.setMaximum(65535)

        # Add Widgets to Layouts
        # ----------------------
        self.addWidget(playback_widget)
        self.addSeparator()
        self.addWidget(playback_speed_widget)
        self.addSeparator()
        self.addWidget(frame_range_widget)
        self.addSeparator()
        self.addWidget(self.current_frame_spin_box)

class PlayerWidget(QtWidgets.QWidget):

    TITLE = 'Player'

    image_loading_signal = QtCore.Signal(int)
    image_loaded_signal = QtCore.Signal(int)

    frame_changed = QtCore.Signal(int)
    
    def __init__(self, input_path: str = str(), parent=None):
        super().__init__(parent)
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
        self._is_setting_frame = False

        self.viewer = ImageViewerGLWidget(parent=self)

        self.thread_pool = self._create_thread_pool()

        self.play_forward_timer = QtCore.QTimer(self)
        self.play_backward_timer = QtCore.QTimer(self)

        self.tabler_icon = TablerQIcon(opacity=0.8)

    def __init_ui(self):
        """Initialize the UI of the widget.
        """
        # Setup Main Widget
        # -----------------
        self.setWindowTitle(self.TITLE)
        self.setWindowIcon(TablerQIcon.player_play)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        # Main Layout
        # -----------
        self.main_vertical_layout = QtWidgets.QVBoxLayout(self)
        self.main_vertical_layout.setContentsMargins(0, 0, 0, 0)
        # Viewer
        self.main_vertical_layout.addWidget(self.viewer)
        
        # top_left_widget
        self.top_left_widget = QtWidgets.QWidget(self)
        self.top_left_layout = QtWidgets.QHBoxLayout(self.top_left_widget)
        self.top_left_layout.setContentsMargins(0, 0, 0, 0)

        self.top_left_widget.setGraphicsEffect(DropShadowEffect())

        # prefetch_button
        self.prefetch_button = QtWidgets.QPushButton("Prefetch", self.top_left_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        self.prefetch_button.setSizePolicy(sizePolicy)

        # num_cores_spin_box
        self.num_cores_spin_box = QtWidgets.QSpinBox(self.top_left_widget)
        self.num_cores_spin_box.setButtonSymbols(QtWidgets.QSpinBox.ButtonSymbols.NoButtons)
        self.num_cores_spin_box.setValue(8)

        # cpu_button
        self.cpu_button = QtWidgets.QPushButton(self.tabler_icon.cpu_2, '', self.top_left_widget)
        self.cpu_button.setEnabled(False)
        prefetch_widget = GroupWidget(self.prefetch_button, self.num_cores_spin_box, self.cpu_button)

        # Add widgets to top_left_layout
        self.top_left_layout.addWidget(prefetch_widget)

        # Create Layouts
        # --------------
        self.overlay_layout = QtWidgets.QVBoxLayout(self.viewer)
        self.overlay_top_layout = QtWidgets.QHBoxLayout()
        self.overlay_top_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.overlay_layout.addLayout(self.overlay_top_layout)

        # Create Widgets
        # --------------
        self.view_tool_bar = QtWidgets.QToolBar(self)
        self.view_tool_bar.setGraphicsEffect(DropShadowEffect())
        self.view_tool_bar.setStyleSheet('''
            QToolBar {
                background: transparent;
            }

            QPushButton {
                border-top-right-radius: 0;
                border-bottom-right-radius: 0;
                border-right: 0px;
            }

            QDoubleSpinBox {
                border-top-left-radius: 0;
                border-bottom-left-radius: 0;
            }
        ''')

        self.gain_spin_box_widget = DoubleSpinBoxWidget(
            default_value=1.0, 
            icon=self.tabler_icon.flip.brightness_half, 
            toolTip='Gain',
            parent=self,
        )
        self.gamma_spin_box_widget = DoubleSpinBoxWidget(
            default_value=1.0,
            icon=self.tabler_icon.contrast,
            toolTip='Gamma',
            parent=self,
        )
        self.saturation_spin_box_widget = DoubleSpinBoxWidget(
            default_value=1.0,
            icon=self.tabler_icon.color_filter,
            toolTip='Saturation',
            parent=self,
        )

        # Top Right
        # ---------
        self.view_tool_bar.addWidget(self.gain_spin_box_widget)
        self.view_tool_bar.addWidget(self.gamma_spin_box_widget)
        self.view_tool_bar.addWidget(self.saturation_spin_box_widget)

        self.overlay_top_layout.addWidget(self.top_left_widget, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        self.overlay_top_layout.addWidget(self.view_tool_bar, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        # Bottom
        # ------
        # Controller bar at the bottom
        self.controller_bar = ControllerBarWidget(parent=self)
        self.controller_bar.setGraphicsEffect(DropShadowEffect())
        self.overlay_layout.addWidget(self.controller_bar, alignment=QtCore.Qt.AlignmentFlag.AlignBottom)

        # Add reference
        self.frame_indicator_bar = self.controller_bar.frame_indicator_bar
        self.frame_slider = self.controller_bar.frame_slider
        self.current_frame_spin_box = self.controller_bar.current_frame_spin_box
        self.start_frame_spin_box = self.controller_bar.start_frame_spin_box
        self.end_frame_spin_box = self.controller_bar.end_frame_spin_box

        # Playback Controls
        # -----------------
        self.set_image(self.image)
        self.set_playback_speed()

    def __init_signal_connections(self):
        """Initialize signal-slot connections.
        """
        self.prefetch_button.clicked.connect(self.prefetch)

        self.controller_bar.playback_speed_combo_box.currentTextChanged.connect(self.set_playback_speed)

        self.play_forward_timer.timeout.connect(self.next_frame)
        self.play_backward_timer.timeout.connect(self.previous_frame)

        self.controller_bar.play_forward_button.clicked.connect(self.play_forward)
        self.controller_bar.play_backward_button.clicked.connect(self.play_backward)
        self.controller_bar.stop_button.clicked.connect(self.stop_playback)

        self.gamma_spin_box_widget.valueChanged.connect(self.viewer.set_gamma)
        self.gain_spin_box_widget.valueChanged.connect(self.viewer.set_gain)
        self.saturation_spin_box_widget.valueChanged.connect(self.viewer.set_saturation)

        self.current_frame_spin_box.valueChanged.connect(self._proxy_set_frame)
        self.frame_slider.valueChanged.connect(self._proxy_set_frame)

        self.image_loading_signal.connect(lambda frame_number: self.frame_indicator_bar.update_frame_status(frame_number, FrameStatus.CACHING))
        self.image_loaded_signal.connect(lambda frame_number: self.frame_indicator_bar.update_frame_status(frame_number, FrameStatus.CACHED))

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

        self.frame_indicator_bar.set_frame_range(self.first_frame, self.last_frame)

        self.start_frame_spin_box.setValue(self.first_frame)
        self.end_frame_spin_box.setValue(self.last_frame)

    def prefetch(self):
        # Calculate the range of frames to prefetch
        start_frame = self.current_frame
        end_frame = self.end_frame_spin_box.value()

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

    def _proxy_set_frame(self, frame_number: int):
        if self._is_setting_frame:
            return
        self.set_frame(frame_number)

    def set_frame(self, frame_number: int):
        self._is_setting_frame = True

        self.current_frame = frame_number
        self.current_frame_spin_box.setValue(int(self.current_frame))
        self.frame_slider.setValue(int(self.current_frame))

        self.viewer.set_frame(self.current_frame)
        self.frame_indicator_bar.update_frame_status(self.current_frame, FrameStatus.CACHED)

        self._is_setting_frame = False

        # Emit the frame_changed signal
        self.frame_changed.emit(self.current_frame)

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

class MediaPlayerWidget(QtWidgets.QWidget):
    """A media player widget that includes a playlist with drag-and-drop and paste support."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__init_ui()
        self.__init_signals()
        
    def __init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle('Media Player')
        self.setWindowIcon(TablerQIcon.player_play)
        self.resize(800, 600)
        
        # Main layout
        self.main_layout = QtWidgets.QVBoxLayout(self)
        
        # Import Button
        self.import_button = QtWidgets.QPushButton("Import Media", self)
        self.main_layout.addWidget(self.import_button)
        
        # Playlist and Player Widget
        self.playlist_widget = QListWidgetWithDrop(self)
        self.playlist_widget.setMinimumWidth(200)

        self.playlist_widget.set_fields(SequenceFileUtil.FILE_INFO_FIELDS)
        self.playlist_widget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.playlist_widget.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.playlist_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.playlist_widget.set_image_field('file_path')
        self.playlist_widget.setAcceptDrops(True)
        
        # Player Widget
        self.player_widget = PlayerWidget(parent=self)
        
        # Splitter to resize playlist and player
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.playlist_widget)
        self.splitter.addWidget(self.player_widget)
        self.splitter.setStretchFactor(1, 4)
        
        self.main_layout.addWidget(self.splitter)
        
        # Context Menu for Playlist
        self.playlist_widget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        
    def __init_signals(self):
        """Connect signals and slots."""
        self.import_button.clicked.connect(self.import_media)
        self.playlist_widget.itemDoubleClicked.connect(self.play_selected_media)
        self.playlist_widget.customContextMenuRequested.connect(self.show_playlist_context_menu)
        self.playlist_widget.files_dropped.connect(self.add_media_files)
        self.playlist_widget.paths_pasted.connect(self.add_media_files)

    def import_media(self):
        """Open file dialog to import media and handle sequences."""
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            self.add_media_files(self._group_sequences(selected_files))

    def add_media_files(self, file_paths: List[str]):
        """Add media files or paths to the playlist."""
        for path in file_paths:
            data_dict = SequenceFileUtil.extract_file_info(path)
            self.playlist_widget.add_item(data_dict)

    def play_selected_media(self, item: QtWidgets.QListWidgetItem):
        """Play the selected media item."""
        media_path = item.get_value('file_path')
        image_sequence = ImageSequence(media_path)
        self.player_widget.set_image(image_sequence)
        
    def show_playlist_context_menu(self, position):
        """Show context menu for the playlist."""
        menu = QtWidgets.QMenu()
        remove_action = menu.addAction("Remove")
        action = menu.exec_(self.playlist_widget.viewport().mapToGlobal(position))
        if action == remove_action:
            for item in self.playlist_widget.selectedItems():
                self.playlist_widget.takeItem(self.playlist_widget.row(item))
                
    def _group_sequences(self, file_paths: List[str]) -> List[str]:
        """Group files into sequences if they match a sequence pattern."""
        sequence_dict = {}
        
        for path in file_paths:
            dirname, filename = os.path.split(path)
            base, ext = os.path.splitext(filename)
            
            # Match frame number pattern
            match = re.match(r"(.+?)(\d+)$", base)
            if match:
                prefix, frame = match.groups()
                template = f"{dirname}/{prefix}{'#' * len(frame)}{ext}"
                
                if template not in sequence_dict:
                    sequence_dict[template] = []
                sequence_dict[template].append(path)
            else:
                sequence_dict[path] = [path]
        
        # Sort frame files in each sequence
        grouped_files = []
        for template, files in sequence_dict.items():
            if len(files) > 1:  # If a sequence is detected
                grouped_files.append(template)
            else:
                grouped_files.extend(files)
        
        return grouped_files

class QListWidgetWithDrop(GalleryWidget):
    """QListWidget that accepts drag-and-drop of files and text paths, and supports paste action."""

    files_dropped = QtCore.Signal(list)
    paths_pasted = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        """Handle drag enter event."""
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        """Handle drag move event."""
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)
            
    def dropEvent(self, event):
        """Handle drop event."""
        if event.mimeData().hasUrls():
            file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
            self.files_dropped.emit(self._sanitize_paths(file_paths))
            event.acceptProposedAction()
        elif event.mimeData().hasText():
            text = event.mimeData().text()
            paths = text.strip().splitlines()
            self.files_dropped.emit(self._sanitize_paths(paths))
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
            
    def keyPressEvent(self, event):
        """Handle key press events for paste action."""
        if event.matches(QtGui.QKeySequence.StandardKey.Paste):
            clipboard = QtWidgets.QApplication.clipboard()
            text = clipboard.text()
            if text:
                paths = text.strip().splitlines()
                self.paths_pasted.emit(self._sanitize_paths(paths))
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)
    
    def _sanitize_paths(self, paths: List[str]) -> List[str]:
        """Remove quotes from paths and ensure each path is valid."""
        return [re.sub(r"^[\"']|[\"']$", '', path.strip()) for path in paths]


if __name__ == "__main__":
    from blackboard import theme
    import sys

    app = QtWidgets.QApplication(sys.argv)
    theme.set_theme(app, theme='dark')

    # image_path = 'out/frame.####.exr'
    player_widget = MediaPlayerWidget()
    player_widget.show()

    # Test chnage image
    # image_sequence = ImageSequence(image_path)
    # player_widget.set_image(image_sequence)

    sys.exit(app.exec())
