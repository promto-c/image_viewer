# Third Party Imports
# -------------------
from qtpy import QtCore, QtGui, QtWidgets

class FrameIndicatorBar(QtWidgets.QWidget):
    def __init__(self, total_frames, parent=None):
        super(FrameIndicatorBar, self).__init__(parent)
        self.total_frames = total_frames
        self.frame_status = [0] * total_frames  # Initialize all frames to default (0)
        self.setMinimumHeight(2)  # Adjust based on your needs

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        rect = self.rect()
        frame_width = rect.width() / self.total_frames

        for i, status in enumerate(self.frame_status):
            if status == 0:  # Default
                color = QtGui.QColor('gray')
            elif status == 1:  # Caching
                color = QtGui.QColor('blue')
            elif status == 2:  # Cached
                color = QtGui.QColor('green')

            painter.fillRect(QtCore.QRectF(i * frame_width, 0, frame_width, rect.height()), color)

    def update_frame_status(self, frame_index, status):
        if 0 <= frame_index < self.total_frames:
            self.frame_status[frame_index] = status
            self.update()  # Redraw the widget

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.frame_indicator = FrameIndicatorBar(100)  # Assume 100 frames for this example
        self.setCentralWidget(self.frame_indicator)

        # Example updating frame status
        self.frame_indicator.update_frame_status(5, 1)  # Frame 5 is caching
        self.frame_indicator.update_frame_status(6, 2)  # Frame 6 is cached

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
