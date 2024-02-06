# Third Party Imports
# -------------------
from qtpy import QtCore, QtGui, QtWidgets

class FrameIndicatorBar(QtWidgets.QWidget):
    """Widget to display a bar indicating the status of video frames.
    """
    # Define class-level constants for color representations.
    GRAY_COLOR = QtGui.QColor(29, 29, 29)
    BLUE_COLOR = QtGui.QColor(65, 102, 144)
    GREEN_COLOR = QtGui.QColor(65, 144, 65)

    STATUS_TO_COLOR = {
        'default': GRAY_COLOR,
        'caching': BLUE_COLOR,
        'cached': GREEN_COLOR,
    }

    def __init__(self, total_frames: int, parent=None):
        """Initializes the frame indicator bar with a specified number of frames.

        Args:
            total_frames: An integer specifying the total number of frames.
            parent: The parent widget. Defaults to None.
        """
        super(FrameIndicatorBar, self).__init__(parent)
        self.total_frames = total_frames
        self.frame_status = [0] * total_frames  # Initialize all frames to default (0)
        self.setMinimumHeight(2)  # Adjust based on your needs

    def paintEvent(self, event: QtGui.QPaintEvent):
        """Handles the paint event to draw the frame indicators.

        Args:
            event: The QPaintEvent.
        """
        painter = QtGui.QPainter(self)
        rect = self.rect()

        # Fill the background with the default color
        painter.fillRect(rect, self.GRAY_COLOR)

        frame_width = rect.width() / self.total_frames

        for frame_index, status in enumerate(self.frame_status):
            color = self.STATUS_TO_COLOR.get(status, self.GRAY_COLOR)
            painter.fillRect(QtCore.QRectF(frame_index * frame_width, 0, frame_width, rect.height()), color)

    def update_frame_status(self, frame_index: int, status: str):
        """Updates the status of a specific frame.

        Args:
            frame_index: The index of the frame to update.
            status: A string indicating the new status of the frame.
        """
        if 0 <= frame_index < self.total_frames:
            self.frame_status[frame_index] = status
            # Redraw the widget.
            self.update()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.frame_indicator = FrameIndicatorBar(100)  # Assume 100 frames for this example
        self.setCentralWidget(self.frame_indicator)

        # Example updating frame status
        self.frame_indicator.update_frame_status(5, 'caching')  # Frame 5 is caching
        self.frame_indicator.update_frame_status(6, 'cached')  # Frame 6 is cached

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
