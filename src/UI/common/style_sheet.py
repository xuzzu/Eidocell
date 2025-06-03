# src/UI/common/style_sheet.py
import logging
from enum import Enum

from PySide6.QtCore import QFile, QIODevice, QTextStream
from PySide6.QtWidgets import QApplication, QWidget
from qfluentwidgets import Theme, qconfig


class EidocellStyleSheet(Enum):
    """Style sheet definition for Eidocell components"""

    # Dialogs
    CLASS_CLUSTER_SUMMARY = "class_cluster_summary"
    EXPORT_DIALOG = "export_dialog"
    GALLERY_FILTER_DIALOG = "gallery_filter_dialog"
    SETTINGS_DIALOG = "settings_dialog"
    PROGRESS_DIALOG = "progress_dialog"

    # Navigation Interface - Sessions
    SESSION_MANAGEMENT_MODULE = "session_management_module"

    # Navigation Interface - Workspace Views
    ANALYSIS_CONTROL_PANEL = "analysis_control_panel"
    ANALYSIS_CARD = "analysis_card"
    HISTOGRAM_CONFIG_WIDGET = "histogram_config_widget"
    SCATTER_CONFIG_WIDGET = "scatter_config_widget"
    CLASSES_VIEW_WIDGET = "classes_view_widget"
    CLASS_CARD = "class_card"
    CLASS_TREE_VIEW = "classes_tree_view"
    CLASS_CREATION_CONTROLS = "class_creation_controls"
    CLUSTERS_CONTROL_PANEL = "clusters_control_panel"
    CLUSTERS_CARD = "clusters_card"
    GALLERY_CONTAINER = "gallery_container"
    GALLERY_CONTROLS = "gallery_controls"
    IMAGE_CARD_DELEGATE_STYLES = "image_card_delegate_styles"
    SEGMENTATION_CONTROLS = "segmentation_controls"
    PREVIEW_GRID = "preview_grid"
    ADVANCED_SEGMENTATION_MODULE = "advanced_segmentation_module"
    FLOW_GALLERY = "flow_gallery"
    MAIN_WINDOW_TITLE_BAR = "main_window_title_bar"

    def _get_qss_path(self, theme_mode: Theme):
        """Gets the resource path for the QSS file based on the theme."""
        theme_str = theme_mode.value.lower()
        return f":/eidocell/qss/{theme_str}/{self.value}.qss"

    def apply(self, widget: QWidget, theme: Theme = Theme.AUTO, extra="") -> None:
        current_theme = qconfig.theme if theme == Theme.AUTO else theme
        qss_path = self._get_qss_path(current_theme)

        try:
            file = QFile(qss_path)
            if file.open(QIODevice.ReadOnly | QIODevice.Text):
                stream = QTextStream(file)
                style_sheet_content = stream.readAll()
                file.close()

                widget.setStyleSheet(style_sheet_content + "\n" + extra)
            else:
                logging.warning(
                    f"Could not open QSS file: {qss_path} for widget {widget.objectName() or widget.__class__.__name__}. Error: {file.errorString()}"
                )
        except Exception as e:
            logging.error(
                f"Error applying stylesheet {qss_path} to {widget.objectName() or widget.__class__.__name__}: {e}",
                exc_info=True,
            )

    @classmethod
    def get_global_theme_qss_path(cls, theme=Theme.AUTO):
        theme_str = (
            qconfig.theme.value.lower() if theme == Theme.AUTO else theme.value.lower()
        )
        return f":/eidocell/themes/{theme_str}/global.qss"
