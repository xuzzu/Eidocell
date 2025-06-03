from PySide6.QtCore import Qt
from PySide6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget
from qfluentwidgets import Pivot

from UI.navigation_interface.workspace.views.analysis.analysis_view_widget import (
    AnalysisViewWidget,
)
from UI.navigation_interface.workspace.views.classes.classes_view_widget import (
    ClassesViewWidget,
)
from UI.navigation_interface.workspace.views.clusters.clusters_view_widget import (
    ClustersViewWidget,
)
from UI.navigation_interface.workspace.views.gallery.gallery_view_widget import (
    GalleryViewWidget,
)
from UI.navigation_interface.workspace.views.segmentation.segmentation_view_widget import (
    SegmentationViewWidget,
)


class WorkspaceWidget(QWidget):
    """Widget for the main workspace area, with a pivot to switch between views."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Workspace")
        self.vBoxLayout = QVBoxLayout(self)
        self.pivot = Pivot(self)
        self.stackedWidget = QStackedWidget(self)
        self.galleryView = GalleryViewWidget(parent, self)
        self.classesView = ClassesViewWidget(self)
        self.clustersView = ClustersViewWidget(self)
        self.analysisView = AnalysisViewWidget(parent, parent=self)
        self.segmentationView = SegmentationViewWidget(parent, parent=self)

        self.__initWidget()

    def __initWidget(self):
        """Initialize the layout and add the views to the workspace."""
        self.vBoxLayout.setContentsMargins(30, 20, 30, 30)
        self.vBoxLayout.addWidget(self.pivot, 0, Qt.AlignHCenter)
        self.vBoxLayout.addWidget(self.stackedWidget)

        # Add views to stacked widget and pivot
        self._addSubInterface(self.galleryView, "galleryView", "Gallery")
        self._addSubInterface(self.classesView, "classesView", "Classes")
        self._addSubInterface(self.clustersView, "clustersView", "Clusters")
        self._addSubInterface(self.analysisView, "analysisView", "Analysis")
        self._addSubInterface(self.segmentationView, "segmentationView", "Segmentation")

        # Set initial view
        self.stackedWidget.setCurrentWidget(self.galleryView)
        self.pivot.setCurrentItem(self.galleryView.objectName())

        # Disable by default
        self.setDisabled(True)

        # Connect pivot to stacked widget
        self.pivot.currentItemChanged.connect(
            lambda k: self.stackedWidget.setCurrentWidget(self.findChild(QWidget, k))
        )

    def _addSubInterface(self, widget, objectName, text):
        """Helper function to add views to the stacked widget and pivot."""
        widget.setObjectName(objectName)
        # widget.setAlignment(Qt.AlignCenter)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(routeKey=objectName, text=text)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    workspace = WorkspaceWidget()
    workspace.show()
    app.exec()
