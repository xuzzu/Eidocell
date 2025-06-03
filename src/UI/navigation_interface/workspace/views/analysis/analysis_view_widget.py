# UI/navigation_interface/workspace/views/analysis/analysis_view_widget.py
import logging
import os
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import InfoBar

from UI.dialogs.custom_info_bar import CustomInfoBar, InfoBarPosition
from UI.dialogs.plot_view_widget import PlotViewerWidget
from UI.navigation_interface.workspace.views.analysis.analysis_card import AnalysisCard
from UI.navigation_interface.workspace.views.analysis.chart_configurations.parameter_holders import (
    HistogramParameters,
    ScatterParameters,
)
from UI.navigation_interface.workspace.views.analysis.plot_widget.histogram_chart import (
    HistogramChart,
)
from UI.navigation_interface.workspace.views.analysis.plot_widget.interactive_chart_widget import (
    InteractiveChartWidget,
)
from UI.navigation_interface.workspace.views.analysis.plot_widget.scatter_chart import (
    ScatterChart,
)
from UI.utils.flow_gallery import FlowGallery

# Import the new AnalysisControlPanel
from .analysis_controls import AnalysisControlPanel


class AnalysisViewWidget(QWidget):
    """Widget for the analysis view, displaying charts and segmentation results."""

    def __init__(self, main_window_reference, parent=None):
        super().__init__(parent)
        self.setObjectName("analysisViewWidget")
        self.main_window_reference = main_window_reference
        self.analysis_presenter = None  # Will be set by BackendInitializer
        self.plot_generator = None  # Will be set when presenter is set
        self.analysis_cards: Dict[str, AnalysisCard] = {}
        self.__initWidget()

    def __initWidget(self):
        """Initialize connections and default states."""
        # Main layout is now QHBoxLayout
        self.hBoxLayout = QHBoxLayout(self)
        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)  # No margins for the view itself
        self.hBoxLayout.setSpacing(0)  # No spacing between panel and gallery

        # Left Panel: Analysis Controls
        self.controlPanel = AnalysisControlPanel(self)
        self.hBoxLayout.addWidget(self.controlPanel)

        # Right Panel: Gallery for Charts
        self.gallery = FlowGallery(self)
        self.hBoxLayout.addWidget(self.gallery, 1)

    def set_presenter(self, analysis_presenter):
        self.analysis_presenter = analysis_presenter
        if self.analysis_presenter and self.analysis_presenter.data_manager:
            self.controlPanel.set_data_manager_for_configs(
                self.analysis_presenter.data_manager
            )
        else:
            logging.error(
                "AnalysisPresenter or its DataManager not available when setting presenter for AnalysisView."
            )
            self.plot_generator = None

    def create_interactive_chart_card(
        self,
        plot_id: str,
        plot_config_obj: Dict[str, Any],
        data_manager_for_reads: Any,
        presenter_for_icw: Any,
    ) -> Optional[InteractiveChartWidget]:
        """
        Creates an AnalysisCard containing an InteractiveChartWidget for the given plot configuration.
        Called by AnalysisPresenter.on_plot_added_to_dm.
        Returns the created InteractiveChartWidget instance or None on failure.
        """
        if plot_id in self.analysis_cards:
            logging.warning(
                f"Attempted to create card for existing plot_id: {plot_id}. Ignoring."
            )
            return self.analysis_cards[plot_id].findChild(
                InteractiveChartWidget
            )  # Return existing ICW

        logging.debug(
            f"AnalysisViewWidget: Creating card for plot_id {plot_id}, type: {plot_config_obj['chart_type']}"
        )

        chart_type_key = plot_config_obj["chart_type"]
        # parameters_obj_from_dm is the actual dataclass instance (e.g. HistogramParameters)
        parameters_obj_from_dm = plot_config_obj["parameters_obj"]

        # 1. Instantiate the correct BaseChart derivative
        chart_instance = None
        if chart_type_key == "histogram":
            if not isinstance(parameters_obj_from_dm, HistogramParameters):
                logging.error(
                    f"Plot {plot_id}: Expected HistogramParameters, got {type(parameters_obj_from_dm)}. Cannot create chart."
                )
                return None
            chart_instance = HistogramChart(
                num_bins=parameters_obj_from_dm.num_bins,
                x_axis_label=parameters_obj_from_dm.x_variable,
            )
            chart_instance.set_show_mean(parameters_obj_from_dm.show_mean)
            chart_instance.set_relative_frequency(
                parameters_obj_from_dm.relative_frequency
            )
            # Layered/group_by for histogram not fully implemented in current chart, but params exist
        elif chart_type_key == "scatter":
            if not isinstance(parameters_obj_from_dm, ScatterParameters):
                logging.error(
                    f"Plot {plot_id}: Expected ScatterParameters, got {type(parameters_obj_from_dm)}. Cannot create chart."
                )
                return None
            chart_instance = ScatterChart(
                x_axis_label=parameters_obj_from_dm.x_variable,
                y_axis_label=parameters_obj_from_dm.y_variable,
            )
            chart_instance.set_style(  # Apply styling if parameters exist
                color_by=parameters_obj_from_dm.color_variable,
                trendline=parameters_obj_from_dm.trendline or "none",
                marginal_x=bool(parameters_obj_from_dm.marginal_x),
                marginal_y=bool(parameters_obj_from_dm.marginal_y),
            )
        else:
            logging.error(
                f"Unsupported chart_type_key: {chart_type_key} for plot {plot_id}"
            )
            return None

        # 2. Set data on the chart instance
        # The parameters_obj_from_dm (e.g., HistogramParameters) has a get_data method
        try:
            # Pass data_manager_for_reads (which is self.analysis_presenter.data_manager)
            chart_specific_data = parameters_obj_from_dm.get_data(
                data_manager_for_reads
            )

            if chart_type_key == "histogram":
                data_for_hist_chart = chart_specific_data
                chart_instance.set_data(
                    data_for_hist_chart,
                    value_param_name=parameters_obj_from_dm.x_variable,
                )

            elif chart_type_key == "scatter":
                data_for_scatter_chart = chart_specific_data
                chart_instance.set_data(
                    data_for_scatter_chart,
                    x_param_name=parameters_obj_from_dm.x_variable,
                    y_param_name=parameters_obj_from_dm.y_variable,
                )
        except Exception as e:
            logging.error(
                f"Error getting or setting data for plot {plot_id} (type {chart_type_key}): {e}",
                exc_info=True,
            )
            # Show InfoBar to user
            if self.main_window_reference:
                InfoBar.error(
                    title=f"Data Error for Plot",
                    content=f"Could not load data for {plot_config_obj['name']}: {e}",
                    parent=self.main_window_reference,
                )
            return None

        # 3. Create InteractiveChartWidget with the chart instance
        icw = InteractiveChartWidget(chart_instance)
        icw.plot_id = plot_id  # Assign plot_id

        # 4. Connect signals from ICW to AnalysisPresenter
        # presenter_for_icw is self.analysis_presenter
        icw.new_gate_defined.connect(presenter_for_icw.handle_new_gate_defined_on_plot)
        icw.gate_update_requested.connect(
            presenter_for_icw.handle_gate_update_requested_from_plot
        )
        icw.gate_delete_requested.connect(
            presenter_for_icw.handle_gate_delete_requested_from_plot_table
        )

        # 5. Create AnalysisCard to host the InteractiveChartWidget
        # AnalysisCard's __init__ needs to be updated to accept an ICW.
        card = AnalysisCard(
            plot_id=plot_id,
            interactive_chart_widget_instance=icw,
            plot_name=plot_config_obj["name"],
            parent=self.gallery,
        )
        card.plot_id = plot_id  # Store plot_id on the card for context menu/deletion

        # Connect card signals (e.g., for delete)
        card.deleteRequested.connect(
            lambda p_id=plot_id: presenter_for_icw.handle_delete_plot_card_request(p_id)
        )
        # card.doubleClicked signal might be repurposed or removed if ICW handles all interaction.

        self.gallery.add_item(card)
        self.analysis_cards[plot_id] = card

        logging.info(
            f"AnalysisCard with InteractiveChartWidget created for plot {plot_id}."
        )
        return icw

    def create_analysis_card(self, html_file_path: str):
        """Creates an AnalysisCard and adds it to the gallery."""
        if not os.path.exists(html_file_path):
            logging.error(f"HTML file for chart card not found: {html_file_path}")
            return

        card = AnalysisCard(
            html_file_path, self.gallery
        )  # Parent is gallery for layout
        self.gallery.add_item(card)  # Use FlowGallery's add_item method
        card.doubleClicked.connect(self.open_plot_viewer)
        card.deleteRequested.connect(
            lambda c=card: self.delete_analysis_card(c)
        )  # Ensure correct card is passed

    def open_plot_viewer(self, html_file_path: str):
        """Opens the PlotViewerWidget."""
        absolute_html_path = os.path.abspath(html_file_path)
        # Check if a viewer for this exact path is already open to avoid duplicates
        # This requires managing open viewers, e.g., in a dictionary.
        # For simplicity now, we allow multiple viewers for the same plot.
        plot_viewer = PlotViewerWidget(
            absolute_html_path, self.main_window_reference
        )  # Parent to main window
        plot_viewer.show()

    def delete_analysis_card(self, card: AnalysisCard):
        """Removes the specified AnalysisCard from the layout."""
        self.gallery.remove_item(card)  # Use FlowGallery's remove_item method
        card.deleteLater()
        # Optionally, delete the associated html/png files from "temp" if desired
        try:
            if card.html_file_path and os.path.exists(card.html_file_path):
                os.remove(card.html_file_path)
            # Construct potential png path if you save them consistently
            png_path = card.html_file_path.replace(".html", ".png")
            if os.path.exists(png_path):
                os.remove(png_path)
        except Exception as e:
            logging.warning(f"Could not delete temporary chart file: {e}")

    def remove_analysis_card(self, plot_id: str):
        """Removes the AnalysisCard associated with plot_id from the gallery."""
        card_to_remove = self.analysis_cards.pop(plot_id, None)
        if card_to_remove:
            logging.debug(
                f"AnalysisViewWidget: Removing card for plot_id {plot_id} from gallery."
            )
            self.gallery.remove_item(card_to_remove)  # Assuming FlowGallery.remove_item

            #  Ensure InteractiveChartWidget within the card also clears its state if necessary
            # (Though if AnalysisCard is deleted, its children should be too)
            icw = card_to_remove.findChild(
                InteractiveChartWidget
            )  # Or however ICW is accessed
            if icw and hasattr(
                icw, "clear_internal_state"
            ):  # Add clear_internal_state to ICW if needed
                logging.debug(
                    f"AnalysisViewWidget: Clearing internal state of ICW for plot {plot_id}."
                )
                icw.clear_internal_state()

            card_to_remove.deleteLater()  # Schedule card for deletion
            logging.info(
                f"AnalysisViewWidget: Removed and scheduled deletion for analysis card: {plot_id}"
            )
        else:
            logging.warning(
                f"AnalysisViewWidget: Could not find analysis card for plot_id: {plot_id} to remove."
            )

    def clear_all_plot_cards(self):
        """Removes all plot cards from the gallery and clears internal tracking."""
        logging.info("AnalysisViewWidget: Clearing all plot cards.")
        # Iterate over a copy of keys because remove_analysis_card modifies self.analysis_cards
        for plot_id in list(self.analysis_cards.keys()):
            self.remove_analysis_card(plot_id)

        # Ensure the gallery itself is visually empty
        # FlowGallery.clear_items() might be more direct if it handles child widget deletion
        if hasattr(self.gallery, "clear_items"):
            self.gallery.clear_items()
        else:  # Fallback: iterate and remove (though remove_analysis_card should handle this)
            while self.gallery.flow_layout.count() > 0:
                item = self.gallery.flow_layout.takeAt(0)
                if item and item.widget():
                    item.widget().deleteLater()

        self.analysis_cards.clear()  # Defensive clear
        logging.info("AnalysisViewWidget: All plot cards cleared.")
