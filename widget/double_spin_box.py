# Third Party Imports
# -------------------
from qtpy import QtCore, QtGui, QtWidgets

class AdaptivePaddedDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """A QDoubleSpinBox with adaptive padding and stepping based on cursor position.

    This class extends QDoubleSpinBox to allow dynamic adjustment of display padding
    and step size based on the cursor's position within the spin box.

    Attributes:
        padding_length_before (int): The number of characters to display before the decimal.
        padding_length_after (int): The number of characters to display after the decimal.
    """
    # Initialization and Setup
    # ------------------------
    def __init__(self, padding_length_before: int = 1, padding_length_after: int = 1, 
                 default_value: float = 0.0, min_value: float = 0.0, max_value: float = 99.99, single_step: float = 0.1, parent: QtWidgets.QWidget = None):
        """Initialize the spin box with specific padding lengths.

        Args:
            padding_length_before (int): Initial padding length before the decimal point.
            padding_length_after (int): Initial padding length after the decimal point.
        """
        # Initialize the super class
        super().__init__(parent=parent)

        # Store the arguments
        self.padding_length_before = padding_length_before
        self.padding_length_after = padding_length_after

        self._mouse_press_pos = None
        self._mouse_press_value = None

        self.setRange(min_value, max_value)
        self.setValue(default_value)
        self.setSingleStep(single_step)
        self.lineEdit().installEventFilter(self)

        # Set the maximum number of decimal places
        self.setDecimals(10)

        # Set up signal connections
        self._setup_signal_connections()

    def _setup_signal_connections(self):
        """Set up signal connections between widgets and slots.
        """
        self.lineEdit().cursorPositionChanged.connect(self._adjust_step)
        self.lineEdit().textChanged.connect(self.adjust_padding)

    # Private Methods
    # ---------------
    def _adjust_step(self, old_pos: int, new_pos: int) -> None:
        """Adjust the step size based on the cursor position.

        Args:
            old_pos: The previous position of the cursor (not currently used).
            new_pos: The current position of the cursor.
        """
        # Get the current text from the line edit
        text = self.lineEdit().text()

        # Find the position of the decimal point, or use text length if no decimal point
        decimal_pos = text.find('.') if '.' in text else len(text)

        # Adjust the new cursor position if it's not at the selection start
        if self.lineEdit().selectionStart() != new_pos:
            new_pos -= 1

        # Calculate the distance from the new cursor position to the decimal point
        distance_to_decimal = new_pos - decimal_pos
        distance_to_decimal += 1 if distance_to_decimal < 0 else 0

        # Set the step size based on the distance to the decimal point
        self.setSingleStep(10 ** (-distance_to_decimal))

    # Extended Methods
    # ----------------
    def adjust_padding(self, text: str) -> None:
        """Adjust the padding lengths based on the current text.

        This method updates the padding lengths for both the integer part (before the decimal point)
        and the fractional part (after the decimal point) based on the input text.

        Args:
            text: The current text from the line edit.

        Doctests:
            >>> app = QtWidgets.QApplication(sys.argv)
            >>> spinbox = AdaptivePaddedDoubleSpinBox()

            >>> spinbox.adjust_padding('123')
            >>> spinbox.padding_length_before
            3
            >>> spinbox.padding_length_after
            0
            >>> spinbox.adjust_padding('00123.456')
            >>> spinbox.padding_length_before
            5
            >>> spinbox.padding_length_after
            3
            >>> spinbox.adjust_padding('00123.')
            >>> spinbox.padding_length_before
            5
            >>> spinbox.padding_length_after
            0
        """
        # Split the text at the decimal point to extract integer and decimal parts
        integer_part, _, decimal_part = text.partition('.')

        # Update padding lengths based on the lengths of integer and decimal parts
        self.padding_length_before = len(integer_part)
        self.padding_length_after = len(decimal_part) if decimal_part else 0

    # Event Handling or Override Methods
    # ----------------------------------
    def setValue(self, value: float) -> None:
        """Set the value of the spinbox, adjusting the padding if necessary.

        Args:
            value: The new value to be set for the spinbox. The value should be within the range of the spinbox.
        """
        # Extracts the decimal part of the float value as a string.
        decimal = str(value).split('.')[1] if '.' in str(value) else '' 

        # Update padding to match or exceed the length of the decimal part.
        self.padding_length_after = max(self.padding_length_after, len(decimal))

        # Set the value using the superclass method
        super().setValue(value)

    def textFromValue(self, value: float) -> str:
        """Convert the numeric value to text with proper padding.

        The method formats the given float value to a string that includes leading zeros
        and a fixed number of decimal places. It respects the current padding settings
        for both the integer and fractional parts of the number.

        Args:
            value: The numeric value to convert to a padded string.

        Returns:
            str: A string representation of the value with leading zeros and fixed decimal places.

        Examples:
            >>> app = QtWidgets.QApplication(sys.argv)
            >>> spinbox = AdaptivePaddedDoubleSpinBox()

            >>> spinbox.lineEdit().setText('12.')
            >>> spinbox.textFromValue(12.0)
            '12.'
            >>> spinbox.lineEdit().setText('123')
            >>> spinbox.textFromValue(123.0)
            '123'
            >>> spinbox.lineEdit().setText('0123.460')
            >>> spinbox.textFromValue(123.46)
            '0123.460'
        """
        # Determine the offset for padding, which includes the decimal point if any decimal places are specified.
        if self.padding_length_after:
            offset = self.padding_length_after + 1  # plus one for the decimal point
        else:
            offset = self.padding_length_after  # no decimal point, no offset

        # Format the value to a string with leading zeros and a fixed number of decimal places.
        # The total length includes padding before and after the decimal point, plus the decimal point itself.
        text = "{:0{}.{}f}".format(value, self.padding_length_before + offset, self.padding_length_after)

        # Append a decimal point to the text if the current text in the line edit ends with a decimal point.
        # This maintains the user's input style.
        if self.lineEdit().text().endswith('.'):
            text += '.'

        return text

    def stepBy(self, steps: int) -> None:
        """Step the value, maintaining proper cursor and selection positioning.

        Args:
            steps: The number of steps to increment or decrement the value.
        """
        # Capture current cursor position and selection range
        cursor_position = self.lineEdit().cursorPosition()
        selection_start = self.lineEdit().selectionStart()
        selection_end = self.lineEdit().selectionEnd()
        selection_length = selection_end - selection_start

        # Record the text length before the step to detect changes
        text_length_before = len(self.lineEdit().text())

        # Perform the step operation using the superclass method
        super().stepBy(steps)
        
        # Determine the text length after the step to calculate the adjustment needed
        text_length_after = len(self.lineEdit().text())

        # Calculate the length difference due to the step operation
        length_difference = text_length_after - text_length_before

        # Determine the new cursor position and selection start
        new_cursor_position = cursor_position + length_difference if text_length_before != text_length_after else cursor_position
        new_selection_start = selection_start + length_difference if text_length_before != text_length_after else selection_start

        # Set the new cursor position and selection
        if selection_length > 0:
            self.lineEdit().setSelection(new_selection_start, selection_length)
        else:
            self.lineEdit().setCursorPosition(new_cursor_position)

    def eventFilter(self, source: QtWidgets.QLineEdit, event: QtCore.QEvent):
        if source == self.lineEdit():
            if event.type() == QtCore.QEvent.MouseButtonPress and event.buttons() == QtCore.Qt.MiddleButton:
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
        self.spin_box = AdaptivePaddedDoubleSpinBox(
            default_value=self.default_value, 
            min_value=self.min_value, max_value=self.max_value, 
            single_step=self.single_step, parent=self)

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
