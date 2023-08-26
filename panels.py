
from qtpy import QtWidgets
import NodeGraphQt
from tablerqicon import TablerQIcon

class NodeGraph(NodeGraphQt.NodeGraph):

    WINDOW_TITLE = 'Node Graph'

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.widget.setWindowTitle(self.WINDOW_TITLE)
        self.widget.setWindowIcon(TablerQIcon.chart_dots_3)


class NodesTreeWidget(NodeGraphQt.NodesTreeWidget):

    WINDOW_TITLE = 'Nodes'
    
    def __init__(self, parent=None, node_graph=None):
        super().__init__(parent, node_graph)

        self.setWindowTitle(self.WINDOW_TITLE)
        self.setWindowIcon(TablerQIcon.color_swatch)


class NodePropertiesWidget(QtWidgets.QWidget):
    """
    """

    WINDOW_TITLE = 'Node Properties'

    # Initialization and Setup
    # ------------------------
    def __init__(self, parent=None):
        """Initialize the widget and set up the UI, signal connections, and icon.
        """
        # Initialize the super class
        super().__init__(parent)

        # Store the arguments
        

        # Set up the initial attributes
        self._setup_attributes()
        # Set up the UI
        self._setup_ui()
        # Set up signal connections
        self._setup_signal_connections()
        # Set up the icons
        self._setup_icons()

    def _setup_attributes(self):
        """Set up the initial values for the widget.
        """
        # Attributes
        # ----------
        self.tabler_qicon = TablerQIcon()

        # Private Attributes
        # ------------------

    def _setup_ui(self):
        """Set up the UI for the widget, including creating widgets and layouts.
        """
        self.setWindowTitle(self.WINDOW_TITLE)

        # # Create a vertical layout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Set the layout for this widget
        self.setLayout(self.main_layout)



        self.trash_widget = QtWidgets.QWidget()
        self.trash_layout = QtWidgets.QVBoxLayout()
        self.trash_widget.setLayout(self.trash_layout)
        self.trash_widget.setHidden(True)
        # self.main_layout.addWidget(self.trash_widget)

    def _setup_signal_connections(self):
        """Set up signal connections between widgets and slots.
        """
        # Connect signals to slots here
        pass

    def _setup_icons(self):
        """Set the icons for the widgets.
        """
        # Set the icons for the widgets
        self.setWindowIcon(self.tabler_qicon.adjustments)

    # Private Methods
    # ---------------

    # Extended Methods
    # ----------------
    def clear_panel(self):
        item = self.main_layout.takeAt(0)

        if not item:
            return

        widget = item.widget()
        self.trash_layout.addWidget(widget)
        
        # widget.deleteLater()

    def set_node(self, node: NodeGraphQt.BaseNode):
        # Clear the existing panel first
        self.clear_panel()

        if not hasattr(node, 'panel'):
            return

        # Add the new panel to the main layout
        self.main_layout.addWidget(node.panel)

    # Special Methods
    # ---------------

    # Event Handling or Override Methods
    # ----------------------------------


class ObjectList(QtWidgets.QWidget):

    WINDOW_TITLE = 'Object List'
            
    def __init__(self, parent: QtWidgets.QWidget = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setWindowIcon(TablerQIcon.layout_list)


class ObjectProperties(QtWidgets.QWidget):

    WINDOW_TITLE = 'Object Properties'
            
    def __init__(self, parent: QtWidgets.QWidget = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setWindowIcon(TablerQIcon.server_cog)
