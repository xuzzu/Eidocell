from qfluentwidgets import MessageBoxBase, ProgressBar, SubtitleLabel


class ProgressDialog(MessageBoxBase):
    """Dialog for showing progress during long operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading Images")
        self.titleLabel = SubtitleLabel("Creating image tiles...", self)
        self.progressBar = ProgressBar(self)
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)

        self.yesButton.setText("Close")
        self.yesButton.setVisible(False)
        self.cancelButton.setVisible(False)

        self.__initWidget()

    def __initWidget(self):
        """Initialize the layout and widgets within the dialog."""
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.progressBar)
        # self._hBoxLayout.setContentsMargins(0, 0, 0, 0)
        # self._hBoxLayout.setSpacing(0)
        self.widget.setMinimumWidth(400)
        self.widget.setMinimumHeight(200)
        self.setMaximumSize(400, 200)
        print(self.size())
        print(self.widget.size())

    def update_progress(self, value):
        """Updates the progress bar value."""
        self.progressBar.setValue(value)
        if value == 100:
            self.close()
