# Third Party Imports
# -------------------
from qtpy import QtCore, QtGui, QtWidgets, uic

class DoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, default_value: float = 0.0, min_value: float = 0.0, max_value: float = 99.99, single_step: float = 0.1, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._mouse_press_pos = None
        self._mouse_press_value = None

        self.setRange(min_value, max_value)
        self.setValue(default_value)
        self.setSingleStep(single_step)
        self.lineEdit().installEventFilter(self)

    def eventFilter(self, source: QtWidgets.QLineEdit, event: QtCore.QEvent):
        if source == self.lineEdit():
            if event.type() == QtCore.QEvent.MouseButtonPress and event.buttons() == QtCore.Qt.LeftButton:
                self._mouse_press_pos = event.globalPos()
                self._mouse_press_value = self.value()
                return True
            elif event.type() == QtCore.QEvent.MouseMove and self._mouse_press_pos is not None:
                delta = event.globalPos() - self._mouse_press_pos
                delta_x = delta.x() / 100.0  # Sensitivity factor
                new_value = self._mouse_press_value + delta_x * self.singleStep()
                self.setValue(new_value)
                return True
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                self._mouse_press_pos = None
                return True

        return super().eventFilter(source, event)

class DoubleSpinBoxWidget(QtWidgets.QWidget):

    def __init__(self, default_value: float = 0.0, min_value: float = 0.0, max_value: float = 99.99, single_step: float = 0.1, 
                 icon: QtGui.QIcon = None, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.default_value = default_value
        self.icon = icon
        self.min_value = min_value
        self.max_value = max_value
        self.single_step = single_step

        self._setup_attributes()
        self._setup_ui()
        self._setup_signal_connections()

    def _setup_attributes(self):
        self.button = QtWidgets.QPushButton(self.icon, self)
        self.button.setMaximumHeight(22)
        self.button.setDisabled(True)
        self.spin_box = DoubleSpinBox(default_value=self.default_value, min_value=self.min_value, max_value=self.max_value, single_step=self.single_step, parent=self)

        self.valueChanged = self.spin_box.valueChanged

    def _setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.button)
        layout.addWidget(self.spin_box)

        self.setLayout(layout)

    def _setup_signal_connections(self):
        self.spin_box.valueChanged.connect(self._update_button)
        self.button.clicked.connect(self.set_default)

    def _update_button(self, value: float):
        self.button.setDisabled(value == self.default_value)

    def set_default_value(self, default_value):
        self.default_value = default_value

    def set_default(self):
        self.spin_box.setValue(self.default_value)

    def setIcon(self, icon: QtGui.QIcon):
        self.button.setIcon(icon)
