from PySide6.QtCore import QEvent, QObject, Qt, Signal


class ControlHelper(QObject):
    """Helper class to detect Ctrl key presses."""

    ctrl_signal = Signal(bool)

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self.window.installEventFilter(self)

    @property
    def window(self):
        return self._window

    def eventFilter(self, obj, event):
        if obj is self.window:
            if event.type() == QEvent.KeyPress:
                if event.key() == Qt.Key_Control:
                    self.ctrl_signal.emit(True)
            if event.type() == QEvent.KeyRelease:
                if event.key() == Qt.Key_Control:
                    self.ctrl_signal.emit(False)
        return super().eventFilter(obj, event)
