
import sys

from tablerqicon import TablerQIcon
from qtpy import QtWidgets, QtGui, QtCore

from nodes.node import Node
from blackboard.utils.image_utils import ImageSequence

class ReadNodePanel(QtWidgets.QWidget):
    """A PyQt5 widget with a user interface created from a .ui file.
    
    Attributes:
        ...
    """
    # Initialization and Setup
    # ------------------------
    def __init__(self, parent=None):
        """Initialize the widget and set up the UI, signal connections, and icon.

        Args:
            ...
        """
        # Initialize the super class
        super().__init__(parent)

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
        self.tabler_icon = TablerQIcon()

        # Private Attributes
        # ------------------

    def __init_ui(self):
        """Set up the UI for the widget, including creating widgets and layouts.
        """
        # Create widgets and layouts here

        layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(layout)

        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.file_edit = QtWidgets.QLineEdit(self)

        self.open_button = QtWidgets.QPushButton(self, icon=self.tabler_icon.dots_vertical)
        self.open_button.setFixedWidth(12)


        layout.addWidget(self.file_edit)
        layout.addWidget(self.open_button)

    def __init_signal_connections(self):
        """Set up signal connections between widgets and slots.
        """
        # Connect signals to slots here
        
        self.open_button.clicked.connect(self.show_dialog)

    def _setup_icons(self):
        """Set the icons for the widgets.
        """
        pass

    # Private Methods
    # ---------------

    # Extended Methods
    # ----------------
    def show_dialog(self):

        # Generate the filter string
        # NOTE: Tmp
        media_file_types = ' '.join(f'*.{ext}' for ext in ['exr', 'dpx'])
        filter_string = f'Media Files ({media_file_types});;EXR Files (*.exr);;DPX Files (*.dpx);;All Files (*)'

        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Select Media",
            dir=self.file_edit.text(),
            filter=filter_string
        )

        if not file:
            return
        
        self.file_edit.setText(file)  # Set the selected files in the QLineEdit

    # Special Methods
    # ---------------

    # Event Handling or Override Methods
    # ----------------------------------

class ReadNode(Node):
    """
    """

    # unique node identifier.
    __identifier__ = 'nodes.read'

    # initial default node name.
    NODE_NAME = 'Read'

    def __init__(self):
        super().__init__()

        self.add_text_input(name='file', label='file')

        self.panel = ReadNodePanel()
        self.image = None

        self.vector_entities = list()

        self.panel.file_edit.textChanged.connect(self.set_image)

        if self.panel.file_edit.text():
            self.set_image(self.panel.file_edit.text())

    def frame_range(self):
        if not self.image:
            return
        return self.image.frame_range()

    def set_image(self, input_path):
        self.image = ImageSequence(input_path)

    def set_player(self, player):
        self.player = player

    def get_image_data(self, frame):
        if not self.image:
            return

        image_data = self.image.get_image_data(frame)
        return image_data

    def create_panel(self):
        return ReadNodePanel()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # theme.set_theme(app, theme='dark')
    panel = ReadNodePanel()
    panel.show()
    sys.exit(app.exec_())
