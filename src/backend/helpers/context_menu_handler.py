from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from UI.navigation_interface.workspace.views.clusters.clusters_card import ClustersCard
from UI.navigation_interface.workspace.views.gallery.image_card import ImageCard


class ContextMenuHandler:
    def __init__(self, presenter):
        self.presenter = presenter  # Reference to the relevant presenter

    def show_context_menu(self, obj, event):
        """Shows the context menu for the given card."""
        if isinstance(obj, ImageCard):
            menu = QMenu(self.presenter.gallery_view_widget)
            menu.setStyleSheet("QMenu {background-color: #234f4b; color: white;}")
            menu.setWindowFlags(menu.windowFlags() | Qt.NoFocus)
            self._create_gallery_image_menu(obj, menu)
        elif isinstance(obj, ClustersCard):
            menu = QMenu(self.presenter.clusters_view_widget)
            menu.setStyleSheet("QMenu {background-color: #234f4b; color: white;}")
            menu.setWindowFlags(menu.windowFlags() | Qt.NoFocus)
            self._create_clusters_card_menu(obj, menu)
        menu.exec_(event.globalPos())

    def _create_gallery_image_menu(self, image, menu):
        """Creates context menu options for an Image."""
        assign_class_menu = menu.addMenu("Assign Class")
        for class_object in self.presenter.data_manager.classes.values():
            action = QAction(class_object.name, menu)
            action.triggered.connect(
                lambda checked=False, name=class_object.name: self.presenter.perform_class_assignment(
                    name
                )
            )
            assign_class_menu.addAction(action)

    def _create_clusters_card_menu(self, card, menu):
        """Creates context menu options for ClustersCard."""
        # Split Action (only if one cluster is selected)
        if len(self.presenter.selected_card_ids) == 1:
            split_action = QAction("Split Cluster", menu)
            show_summary_action = menu.addAction("Show Summary")
            # Directly connect to the presenter method
            show_summary_action.triggered.connect(
                lambda: self.presenter.show_summary(card.cluster_id)
            )
            split_action.triggered.connect(
                lambda checked=False, cluster_id=card.cluster_id: self.presenter.split_selected_cluster()
            )
            menu.addAction(split_action)

        # Merge Action (only if two or more clusters are selected)
        if len(self.presenter.selected_card_ids) >= 2:
            merge_action = QAction("Merge Clusters", menu)
            # Directly connect to the presenter method
            merge_action.triggered.connect(
                lambda checked=False: self.presenter.merge_selected_clusters(
                    self.presenter.selected_card_ids
                )
            )
            menu.addAction(merge_action)

        assign_class_menu = menu.addMenu("Assign Class")
        classes = self.presenter.data_manager.classes
        for class_id, class_object in classes.items():
            class_name = class_object.name
            action = assign_class_menu.addAction(class_name)
            # Directly connect to the presenter method
            action.triggered.connect(
                lambda checked=False, name=class_name: self.presenter.assign_clusters_to_class(
                    name
                )
            )
