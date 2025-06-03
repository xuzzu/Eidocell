# frontend/views/gallery_container.py
import logging

from PySide6.QtCore import QModelIndex, QSize, Qt, Slot
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QListView,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from backend.presenters.gallery_model import GalleryModel
from UI.common.style_sheet import EidocellStyleSheet
from UI.navigation_interface.workspace.views.gallery.gallery_delegate import (
    GalleryDelegate,
)


class GalleryView(QListView):
    """
    Custom QListView to display gallery cards efficiently.
    """

    def __init__(self, parent=None):
        """
        Initializes the GalleryView.

        Parameters:
            parent (QWidget): Parent widget.
        """
        super().__init__(parent)

        # Initialize the model
        self.model = GalleryModel(parent=self)
        self.setModel(self.model)

        # Initialize the delegate
        self.delegate = GalleryDelegate(self)
        self.setItemDelegate(self.delegate)

        # Configure the view
        self.setViewMode(QListView.IconMode)
        self.setFlow(QListView.LeftToRight)
        self.setResizeMode(QListView.Adjust)
        self.setSpacing(10)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QListView.ExtendedSelection)
        self.setMovement(QListView.Static)
        self.setWrapping(True)
        self.setMouseTracking(True)
        self.setLayoutMode(QListView.Batched)
        self.setBatchSize(50)

        # Enable smooth scrolling for better performance
        self.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.setHorizontalScrollMode(QListView.ScrollPerPixel)

        self.delegate.card_size_changed.connect(
            self.handle_card_size_changed_by_delegate
        )

        # Enable drag and drop if needed
        # self.setDragEnabled(True)
        # self.setAcceptDrops(True)
        # self.setDropIndicatorShown(True)

        # Optional: Set minimum size
        self.setMinimumSize(200, 200)

        # Remove default border and set transparent background
        self.setStyleSheet(
            """
            QListView {
                border: none;
                background-color: transparent;
            }
        """
        )

        # Optional: Adjust the palette to ensure transparency
        palette = self.palette()
        palette.setColor(QPalette.Base, QColor(0, 0, 0, 0))  # Transparent base
        self.setPalette(palette)
        self.setAutoFillBackground(False)

    @Slot(QSize)
    def handle_card_size_changed_by_delegate(self, new_size: QSize):
        """
        Called when the delegate's card_size changes.
        This method ensures the model invalidates its pixmap cache for re-scaling
        and the view re-layouts its items.
        """
        logging.debug(f"GalleryView: Handling card_size_changed to {new_size}")
        if self.model:
            # Tell the model to clear its pre-scaled pixmap cache.
            # This will cause new pixmaps to be loaded and scaled to the new_size.
            self.model.invalidate_pixmap_cache_and_refresh()

        # Force the QListView to recalculate the layout of its items.
        # This is necessary because the sizeHint for items has changed.
        self.doItemsLayout()

        # Ensure the viewport is updated/repainted.
        self.viewport().update()
        logging.debug(
            f"GalleryView: Items re-layout triggered and viewport updated for size {new_size}"
        )

    def contextMenuEvent(self, event):
        index = self.indexAt(event.pos())
        if index.isValid():
            image = index.data(Qt.UserRole)  # Retrieve the Image object
            # Access the context menu handler from the presenter
            print(self.parent())
            context_menu_handler = (
                self.parent().parent().parent().gallery_presenter.context_menu_handler
            )  # accessing main_frame -> gallery_container -> gallery_view_widget -> gallery_presenter
            context_menu_handler.show_context_menu(image, event)
        else:
            # Optionally handle context menu for empty space
            pass


class GalleryContainer(QWidget):
    """
    A container widget that styles the GalleryView with rounded corners and a subtle shadow.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Apply Stylesheet to the GalleryContainer itself
        EidocellStyleSheet.GALLERY_CONTAINER.apply(self)

        # Create the main frame
        self.main_frame = QFrame(self)
        self.main_frame.setObjectName("galleryContainerFrame")

        # Apply a subtle shadow effect to the frame
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)  # Controls the blur radius of the shadow
        shadow.setXOffset(0)  # Horizontal offset
        shadow.setYOffset(0)  # Vertical offset
        shadow.setColor(QColor(0, 0, 0, 20))  # Shadow color with transparency
        self.main_frame.setGraphicsEffect(shadow)

        # Set up the layout for the frame
        frame_layout = QVBoxLayout(self.main_frame)
        frame_layout.setContentsMargins(15, 15, 15, 15)  # Inner margins
        frame_layout.setSpacing(10)  # Space between widgets inside the frame

        # Initialize the GalleryView
        self.gallery_view = GalleryView(self.main_frame)
        self.gallery_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Optional: Adjust GalleryView's properties if needed
        # e.g., self.gallery_view.setMinimumHeight(300)

        # Add the GalleryView to the frame's layout
        frame_layout.addWidget(self.gallery_view)

        # Set up the main layout for the container
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.main_frame)
        main_layout.setContentsMargins(10, 10, 10, 10)  # Outer margins
        self.setLayout(main_layout)

        # Optionally, set a minimum size to prevent squishing
        self.setMinimumSize(400, 300)  # Adjust as needed
