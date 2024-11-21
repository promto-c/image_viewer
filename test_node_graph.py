import sys, os
os.environ['QT_API'] = 'PyQt5'
from typing import TYPE_CHECKING

from qtpy import QtCore, QtGui, QtWidgets
from tablerqicon import TablerQIcon

from blackboard import theme

from nodes.tracker_node.tracker_node import TrackerNode
from nodes.read_node import ReadNode
from panels import NodeGraph, NodePropertiesWidget, ObjectList, ObjectProperties, NodesTreeWidget
from player import PlayerWidget

class DockTitleBar(QtWidgets.QWidget):
    def __init__(self, title: str = 'Title', icon: QtGui.QIcon = None, parent=None):
        super().__init__(parent)

        self.icon_button = QtWidgets.QToolButton(icon=icon, parent=self)
        self.title_label = QtWidgets.QLabel(title, self)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.icon_button)
        layout.addWidget(self.title_label)

        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        # Set the background color to match the WindowTitle color role
        palette = self.palette()
        color = palette.color(QtGui.QPalette.ColorRole.Base)
        self.setAutoFillBackground(True)
        palette.setColor(self.backgroundRole(), color)
        self.setPalette(palette)
        
    def set_icon(self, icon: QtGui.QIcon):
        self.icon_button.setIcon(icon)

    def set_title(self, title):
        self.title_label.setText(title)

class DockWidget(QtWidgets.QDockWidget):
    def __init__(self, widget: QtWidgets.QWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setParent(widget.parent())

        window_title = widget.windowTitle()
        self.setWindowTitle(window_title)

        title_bar = DockTitleBar(window_title, widget.windowIcon(), self)

        self.setTitleBarWidget(title_bar)
        self.setWidget(widget)

class DockContainer(QtWidgets.QMainWindow):
    ...

# TODO: Use DockContainer to handle docks then turn this class to QWidget
class MainWindow(QtWidgets.QMainWindow):
    """A PyQt5 main window for a VFX Composite App.
    
    Attributes:
        ...
    """
    def __init__(self, parent=None):
        """Initialize the main window and set up the UI, signal connections, and menu."""
        super().__init__(parent)
        
        self.__init_attributes()
        self.__init_ui()
        self.__init_signal_connections()

    def __init_attributes(self):
        """Set up the initial values and widgets for the main window."""
        self.tabler_qicon =  TablerQIcon()

        self.player = PlayerWidget(parent=self)
        
        self.node_graph = NodeGraph(self)
        self.node_graph.set_layout_direction(1)

        # registered example nodes.
        self.node_graph.register_nodes([
            TrackerNode,
            ReadNode,
        ])

        self.nodes_tree = NodesTreeWidget(parent=self, node_graph=self.node_graph)
        self.node_properties = NodePropertiesWidget(self)
        self.object_list = ObjectList()
        self.object_properties = ObjectProperties()

    def __init_ui(self):
        """Set up the UI for the main window, including creating widgets and layouts."""
        self.setWindowTitle("VFX Composite App")
        self.setDockNestingEnabled(True)

        # Dock
        # ----
        docks = [
            self.nodes_tree,
            self.node_graph.widget,
            self.node_properties,
            self.player,
            self.object_list,
            self.object_properties,
        ]

        for widget in docks:
            self._add_dock(widget)

        # Menu
        # ----
        # self.menu = self.menuBar().addMenu("&View")
        # self.add_viewer_action = QtWidgets.QAction("Add Viewer", self)
        # self.menu.addAction(self.add_viewer_action)

    def __init_signal_connections(self):
        """Set up signal connections between widgets and slots."""
        # Any specific signal connections can be added here.
        # self.add_viewer_action.triggered.connect(self._add_viewer)
        self.node_graph.node_selected.connect(self.node_properties.set_node)
        self.node_graph.node_double_clicked.connect(self.player.set_image)

    # def _add_viewer(self):
    #     """Slot to add a new viewer when triggered."""
    #     viewer_demo_widget = QtWidgets.QTextEdit()
    #     self._add_dock(viewer_demo_widget)

    def _add_dock(self, widget: QtWidgets.QWidget) -> QtWidgets.QDockWidget:
        """Add a dock widget with the given widget and title.
        
        Args:
            widget (QtWidgets.QWidget): The widget to be added to the dock.
        
        Returns:
            QtWidgets.QDockWidget: The created dock widget.
        """
        dock_widget = DockWidget(widget, self)
        
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, dock_widget)
        return dock_widget

def main():
    """Create the application and main window, and show the widget."""
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    theme.set_theme(app, theme='dark')
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
