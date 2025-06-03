### UI/navigation_interface/workspace/views/clusters/clusters_view_widget.py
import logging
from typing import Optional

from PySide6.QtWidgets import QHBoxLayout, QWidget

from UI.navigation_interface.workspace.views.clusters.clusters_card import ClustersCard
from UI.navigation_interface.workspace.views.clusters.clusters_controls import (
    ControlPanel,
)
from UI.utils.flow_gallery import FlowGallery


class ClustersViewWidget(QWidget):
    """Widget to display and manage image clusters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clusters = []  # List to store ClustersCard objects
        self.hBoxLayout = QHBoxLayout(self)
        self.controlPanel = ControlPanel(self)
        self.clustersGallery = FlowGallery(self)
        self.clustersGallery.flow_layout.needAni = True
        self.clustersGallery.flow_layout.duration = 150
        self.clusters_presenter = None  # Initialize the presenter
        self.cluster_index_map = {}
        self.__initWidget()

    def __initWidget(self):
        """Initialize the layout and appearance of the clusters view."""
        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.hBoxLayout.addWidget(self.controlPanel)
        self.hBoxLayout.addWidget(self.clustersGallery, 1)  # Gallery stretches

    def create_cluster_card(self, cluster_id, preview_image_path):
        """Creates and adds a new ClustersCard to the gallery."""
        if cluster_id not in self.cluster_index_map:
            self.cluster_index_map[cluster_id] = len(self.cluster_index_map) + 1
        display_index = self.cluster_index_map[cluster_id]

        shortened_id = cluster_id[:8]  # Shorten the cluster ID
        card = ClustersCard(preview_image_path, self)
        card.clusters_presenter = self.clusters_presenter
        card.cluster_id = cluster_id
        card.label.setText(str(display_index))  # Set the shortened ID as the card label
        self.clusters.append(card)
        self.clustersGallery.flow_layout.addWidget(card)
        return card

    def clear_cluster_cards(self, cluster_ids=None):
        """
        Clears cluster cards from the gallery.

        Args:
            cluster_ids: List of cluster IDs to clear. If None, all clusters are cleared.
        """
        if not cluster_ids:  # Clear all clusters
            for i in reversed(range(self.clustersGallery.flow_layout.count())):
                item = self.clustersGallery.flow_layout.takeAt(
                    i
                )  # Remove the layout item
                item.deleteLater()
            for card in self.clusters:
                card.deleteLater()
            self.clusters = []  # Clear the list of cards
            self.cluster_index_map.clear()
        else:  # Clear specific clusters
            for cluster_id in cluster_ids:
                for card in self.clusters:
                    if card.cluster_id == cluster_id:
                        self.clustersGallery.flow_layout.removeWidget(card)
                        card.deleteLater()
                        self.clusters.remove(card)
                    if cluster_id in self.cluster_index_map:
                        del self.cluster_index_map[cluster_id]
            self._reindex_clusters()

    def _reindex_clusters(self):
        """Reassigns display indices to clusters."""
        new_index_map = {}
        current_index = 1
        for (
            cluster_id
        ) in (
            self.clusters_presenter.data_manager.clusters
        ):  # Use data_manager to get the actual clusters
            new_index_map[cluster_id] = current_index
            current_index += 1

        self.cluster_index_map = new_index_map

        # Update card labels to reflect the new indices
        for card in self.clusters:
            display_index = self.cluster_index_map.get(card.cluster_id)
            if display_index is not None:  # Only update if the cluster still exists
                card.label.setText(str(display_index))
                card.update()  # force refresh

    def set_presenter(self, presenter):
        """Sets the ClustersPresenter for this widget."""
        self.clusters_presenter = presenter

    def clear_clusters(self) -> None:
        """Clears all cluster cards from the gallery."""
        logging.info("Clearing all cluster cards from ClustersViewWidget.")
        for card in self.clusters:
            self.clustersGallery.flow_layout.removeWidget(card)
            card.deleteLater()
        self.clusters.clear()
        logging.info("All cluster cards cleared from ClustersViewWidget.")

    def remove_cluster_card(self, cluster_id: str):
        """Removes a specific cluster card from the gallery."""
        card_to_remove = None
        for card in self.clusters:
            if card.cluster_id == cluster_id:
                card_to_remove = card
                break
        if card_to_remove:
            self.clustersGallery.flow_layout.removeWidget(
                card_to_remove
            )  # Assuming GalleryWidget has remove_card
            self.clusters.remove(card_to_remove)
            card_to_remove.deleteLater()
            logging.info(f"Removed cluster card: {cluster_id}")
        else:
            logging.warning(f"Could not find cluster card {cluster_id} to remove.")

    def find_cluster_card(self, cluster_id: str) -> Optional[ClustersCard]:
        """Finds a cluster card by its ID."""
        for card in self.clusters:
            if card.cluster_id == cluster_id:
                return card
        return None
