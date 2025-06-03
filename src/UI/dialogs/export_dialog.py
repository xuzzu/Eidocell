# UI/dialogs/export_dialog.py

import logging

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QFileDialog, QGridLayout, QLabel, QSizePolicy, QSpacerItem
from qfluentwidgets import CheckBox
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import MessageBoxBase, PushButton, SubtitleLabel

from UI.common.style_sheet import EidocellStyleSheet


class ExportDialog(MessageBoxBase):
    """Dialog for exporting session data with improved visibility and layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Data")

        # Apply Stylesheet
        EidocellStyleSheet.EXPORT_DIALOG.apply(self)

        # Initialize widgets
        self.descriptionLabel = SubtitleLabel("Export Options", self)
        self.descriptionLabel.setObjectName("descriptionLabel")

        self.chooseFolderButton = PushButton(FIF.FOLDER_ADD, "Choose Folder", self)
        self.chooseFolderButton.setObjectName("chooseFolderButton")

        self.folderPathLabel = QLabel("No folder selected", self)
        self.folderPathLabel.setObjectName("folderPathLabel")  # < SET OBJECTNAME

        self.includeClustersCheckBox = CheckBox("Include Clusters", self)
        self.includeMasksCheckBox = CheckBox("Include Masks", self)
        self.includeCalculatedParamsCheckBox = CheckBox(
            "Include Calculated Parameters (.csv)", self
        )

        self.__initWidget()

    def __initWidget(self):
        """Initialize the layout and widgets within the dialog."""
        layout = self.viewLayout  # Use the existing layout from MessageBoxBase
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Add description label
        layout.addWidget(self.descriptionLabel, alignment=Qt.AlignCenter)

        # Create a grid layout for the folder selection and options
        gridLayout = QGridLayout()
        gridLayout.setSpacing(10)
        gridLayout.setContentsMargins(0, 0, 0, 0)

        # Folder Selection
        gridLayout.addWidget(self.chooseFolderButton, 0, 0, 1, 1)
        gridLayout.addWidget(
            self.folderPathLabel, 0, 1, 1, 2, Qt.AlignLeft | Qt.AlignVCenter
        )

        # Export Options
        gridLayout.addWidget(self.includeClustersCheckBox, 1, 0, 1, 1)
        gridLayout.addWidget(self.includeMasksCheckBox, 1, 1, 1, 2)
        gridLayout.addWidget(self.includeCalculatedParamsCheckBox, 2, 1, 1, 2)

        # Add grid layout to the main layout
        layout.addLayout(gridLayout)

        # Spacer to push buttons to the bottom
        layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # Set button texts
        self.yesButton.setText("Export")
        self.cancelButton.setText("Cancel")

        # Adjust the dialog size for better visibility
        self.widget.setFixedWidth(500)  # Increased width from 420 to 500
        self.widget.setFixedHeight(350)  # Increased height from 300 to 350

        # Adjust the "Choose Folder" button size
        self.chooseFolderButton.setFixedHeight(40)
        self.chooseFolderButton.setFixedWidth(150)  # Fixed width for consistency

        # Connect the "Choose Folder" button to the folder picker
        self.chooseFolderButton.clicked.connect(self.select_export_folder)

    def select_export_folder(self):
        """Opens a folder dialog to select the export folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Export Folder", "", QFileDialog.ShowDirsOnly
        )
        if folder_path:
            self.folderPathLabel.setText(folder_path)
        else:
            self.folderPathLabel.setText("No folder selected")

    @Slot()
    def handle_export(self):
        """Handle the export button click."""
        selected_folder = self.folderPathLabel.text()
        if selected_folder == "No folder selected":
            logging.warning("Export attempted without selecting a folder.")
            self.warning_message(
                "No Folder Selected", "Please choose a folder to export the data."
            )
            return

        # Gather selected options
        options = {
            "Include Clusters": self.includeClustersCheckBox.isChecked(),
            "Include Masks": self.includeMasksCheckBox.isChecked(),
            "Include Charts": self.includeChartsCheckBox.isChecked(),
            "Include Calculated Parameters (.csv)": self.includeCalculatedParamsCheckBox.isChecked(),
        }

        logging.info(
            f"Export initiated to folder: {selected_folder} with options: {options}"
        )

        # Emit the export_requested signal with the folder path and options
        self.export_requested.emit(selected_folder, options)

        # Accept the dialog after emitting the signal
        self.accept()

    def warning_message(self, title, message):
        """Displays a warning message to the user."""
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(self, title, message, QMessageBox.Ok)
