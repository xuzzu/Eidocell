import logging

from PySide6.QtCore import QSize, QTimer, Slot
from PySide6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import InfoBar, InfoBarIcon, InfoBarPosition

from UI.navigation_interface.workspace.views.gallery.gallery import GalleryContainer
from UI.navigation_interface.workspace.views.gallery.gallery_controls import (
    GalleryControls,
)


class GalleryViewWidget(QWidget):
    def __init__(self, main_window_reference, parent=None):
        super().__init__(parent)
        self.gallery_presenter = None
        self.cards = []
        self.main_layout = QVBoxLayout(self)
        self.controls = GalleryControls(self)
        self.gallery_container = GalleryContainer(self)  # Placeholder for Gallery
        self.gallery_container.installEventFilter(self)
        self.main_layout.addWidget(self.controls)
        self.main_layout.addWidget(self.gallery_container, 1)  # Gallery will stretch
        self.main_layout.setSpacing(0)
        self.main_window_reference = main_window_reference
        self.loading_more = False

        # Timer to check scroll area position
        self.scroll_check_timer = QTimer(self)  # Timer to check scrollbar value
        self.scroll_check_timer.timeout.connect(self.check_scroll_and_load)

        # Parameter states
        self.sorting_order = "Ascending"
        self.sorting_parameter = "Area"

        # # Connect signals
        # self.controls.scale_slider.valueChanged.connect(self.resize_tiles)
        # self.resize_tiles(self.controls.scale_slider.value())  # Set initial tile size
        # self.controls.sortAscButton.toggled.connect(self.updateSortingOrder)
        # self.controls.sortDescButton.toggled.connect(self.updateSortingOrder)
        # self.controls.parameterComboBox.currentIndexChanged.connect(self.updateSortingParameter)

    def set_presenter(self, gallery_presenter):
        self.gallery_presenter = gallery_presenter

    def handle_gallery_mouse_press(self, event):
        """Handles mouse press events on the gallery."""
        pass

    def get_card_from_widget(self, widget):
        """Traverses up the widget hierarchy to find the GalleryCard."""
        pass

    def handle_gallery_context_menu(self, event):
        """Handles context menu events on the gallery."""
        pass

    def reset_ui_elements(self):
        """Resets the gallery UI elements to their default state."""
        self.controls.mask_toggle.setChecked(False)  # Reset mask view
        self.controls.sortAscButton.setChecked(
            False
        )  # Reset sort order (ascending by default)
        self.controls.sortDescButton.setChecked(False)
        self.controls.parameterComboBox.setCurrentIndex(
            0
        )  # Reset sorting parameter ("Area" by default)
        self.sorting_order = "Ascending"  # Reset the actual sorting order
        self.sorting_parameter = "Area"  # Reset the actual sorting parameter
        self.gallery_presenter.mask_view_enabled = False
        if self.gallery_presenter:
            self.gallery_presenter.toggle_mask_view()  # Update mask view after loading

    def updateSortingOrder(self, checked):
        """Update the sorting order based on the toggled button."""
        if checked:  # Only process when the button is checked
            order = (
                "Ascending"
                if self.sender() is self.controls.sortAscButton
                else "Descending"
            )
            self.sorting_order = order  # Reset sorting order
            print(f"Sorting order changed to: {order}")
            if order == "Ascending":
                self.controls.sortDescButton.setChecked(False)
            else:
                self.controls.sortAscButton.setChecked(False)
            self.gallery_presenter.sort_gallery()

    def updateSortingParameter(
        self, new_parameter_text: str
    ):  # Parameter is the text itself
        """Update the current sorting parameter."""
        # 'new_parameter_text' is the actual string of the selected item
        self.sorting_parameter = new_parameter_text
        logging.info(
            f"GalleryViewWidget: Sorting parameter changed to: {self.sorting_parameter}"
        )  # Use logging
        if self.gallery_presenter:  # Ensure presenter is set
            self.gallery_presenter.sort_gallery()
        else:
            logging.warning(
                "GalleryViewWidget: GalleryPresenter not set, cannot trigger sort_gallery."
            )

    def resize_tiles(self, new_size):
        """Resizes the gallery tiles based on the slider value."""
        new_width = 100 * new_size / 100  # Scale the width based on the slider value
        new_height = int(
            new_width * 1.3
        )  # Maintain aspect ratio (e.g., height = width * 1.3)

        # Update the delegate's card size

        # # Update the delegate's card size
        # self.gallery_container.gallery_view.delegate.set_card_size(QSize(new_width, new_height))

        # # Update the grid size of the view
        # grid_size = QSize(new_width + self.gallery_container.gallery_view.spacing(), new_height + self.gallery_container.gallery_view.spacing())
        # self.gallery_container.gallery_view.setGridSize(grid_size)

        # # Trigger a relayout
        # self.gallery_container.gallery_view.doItemsLayout()
        # self.gallery_container.gallery_view.viewport().update()

        # Access delegate through gallery_container and gallery_view
        delegate = self.gallery_container.gallery_view.delegate
        delegate.set_card_size(QSize(new_width, new_height))

    def on_sorting_failed(self):
        """Handles the sorting failure event."""
        self.reset_ui_elements()
        self.progress_info_bar = InfoBar.new(
            icon=InfoBarIcon.WARNING,
            title="Sorting Failed",
            content="Some images are missing masks. Please ensure all images have associated masks.",
            position=InfoBarPosition.BOTTOM_RIGHT,
            parent=self.main_window_reference,
        )

    def clear_cards(self) -> None:
        """Clears all gallery cards from the view."""
        self.gallery_container.gallery_view.model.clear()

    def update_gallery_layout(self):
        """Updates the gallery layout after sorting."""
        pass

    def check_scroll_and_load(self):
        """Checks the scrollbar position and loads more cards if needed."""
        pass

    @Slot()
    def load_more_cards(self):
        """Loads more cards from the queue."""
        pass
