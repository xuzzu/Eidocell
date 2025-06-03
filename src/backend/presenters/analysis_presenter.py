# backend/presenters/analysis_presenter.py
import dataclasses
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import QMessageBox
from qfluentwidgets import InfoBar, InfoBarIcon, InfoBarPosition

from backend.data_manager import DataManager
from backend.data_manager_interface import DataManagerInterface
from backend.segmentation import SegmentationModel
from UI.navigation_interface.workspace.views.analysis.analysis_view_widget import (
    AnalysisViewWidget,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AnalysisPresenter(QObject):

    def __init__(
        self,
        analysis_view_widget: AnalysisViewWidget,
        data_manager_interface: DataManagerInterface,
        data_manager: DataManager,
        segmentation_model: SegmentationModel,
    ):
        super().__init__()
        self.analysis_view_widget = analysis_view_widget
        self.data_manager_interface = data_manager_interface
        self.data_manager = data_manager
        self.segmentation_model = segmentation_model

        # Store references to InteractiveChartWidgets by plot_id
        self.interactive_chart_widgets: Dict[str, Any] = (
            {}
        )  # Key: plot_id, Value: InteractiveChartWidget instance

        self._connect_ui_signals()

    def _connect_ui_signals(self):
        if self.analysis_view_widget and self.analysis_view_widget.controlPanel:
            # Connect the signal from AnalysisControlPanel to the presenter's handler
            self.analysis_view_widget.controlPanel.create_chart_requested.connect(
                self.handle_create_chart_request_from_ui
            )
        else:
            logging.error(
                "AnalysisPresenter: AnalysisViewWidget or its ControlPanel not found for signal connection."
            )

    @Slot(
        str, object
    )  # chart_type_key (e.g., "histogram"), parameters_obj (dataclass instance)
    def handle_create_chart_request_from_ui(
        self, chart_type_key: str, parameters_obj: object
    ):
        """Handles the request to create a chart from the AnalysisControlPanel."""
        if not self.data_manager_interface:
            logging.error("DataManagerInterface not available in AnalysisPresenter.")
            InfoBar.error(
                title="Error",
                content="Cannot create chart: Backend service unavailable.",
                parent=self.analysis_view_widget.window(),
            )
            return

        # check if all masks are loaded
        for (
            sample_obj
        ) in self.data_manager.active_samples.values():  # Iterate active_samples
            if not (
                sample_obj.mask_id and sample_obj.mask_id in self.data_manager.masks
            ):
                InfoBar.error(
                    title="Mask Previews Required",
                    content="Please run segmentation to have .",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.BOTTOM_RIGHT,
                    duration=4000,
                    parent=self.analysis_view_widget.window(),
                )
                logging.warning("Tried creating the chart without masks, aborting..")
                return

        plot_id = str(uuid.uuid4())

        # Generate a default plot name
        plot_name = (
            f"Plot {len(self.data_manager.plot_configurations) + 1} ({chart_type_key})"
        )
        if hasattr(parameters_obj, "x_variable"):
            plot_name = parameters_obj.x_variable
            if hasattr(parameters_obj, "y_variable") and parameters_obj.y_variable:
                plot_name = (
                    f"{parameters_obj.y_variable} vs {parameters_obj.x_variable}"
                )
            elif chart_type_key == "histogram":
                plot_name = f"Histogram of {parameters_obj.x_variable}"

        if dataclasses.is_dataclass(parameters_obj) and not isinstance(
            parameters_obj, type
        ):
            parameters_dict_to_emit = dataclasses.asdict(parameters_obj)
        elif isinstance(
            parameters_obj, dict
        ):  # Should not happen if control panel sends dataclass
            parameters_dict_to_emit = parameters_obj
        else:
            logging.error(
                f"AnalysisPresenter: parameters_obj is not a dataclass or dict. Type: {type(parameters_obj)}. Cannot proceed."
            )
            InfoBar.error(
                title="Error",
                content="Invalid chart parameters.",
                parent=self.analysis_view_widget.window(),
            )
            return

        logging.info(
            f"AnalysisPresenter: Requesting creation of plot '{plot_name}' (ID: {plot_id}), Type: {chart_type_key}"
        )

        # Pass the actual parameters_obj (dataclass instance)
        self.data_manager_interface.create_plot_request(
            plot_id, plot_name, chart_type_key, parameters_obj
        )
        # UI card creation will be triggered by on_plot_added_to_dm

    def load_plots_and_gates(self):
        """
        Called by BackendInitializer during session load.
        Populates the AnalysisView with existing plots and their gates.
        """
        if not self.data_manager:
            logging.error(
                "AnalysisPresenter: DataManager not available for loading plots/gates."
            )
            return

        logging.info(
            f"AnalysisPresenter: Loading initial plots ({len(self.data_manager.plot_configurations)}) and gates ({len(self.data_manager.gates)})."
        )

        self.interactive_chart_widgets.clear()

        # 1. Create plot cards and their InteractiveChartWidgets
        for plot_id, plot_config_dict in self.data_manager.plot_configurations.items():
            self.on_plot_added_to_dm(plot_id, plot_config_dict)

        for gate_id, gate_obj in self.data_manager.gates.items():
            self.on_gate_added_to_dm(gate_obj.plot_id, gate_obj)

        logging.info("AnalysisPresenter: Finished loading initial plots and gates.")

    @Slot(str, object)  # plot_id, plot_config_obj (dict or PlotConfiguration-like)
    def on_plot_added_to_dm(self, plot_id: str, plot_config_obj: Dict[str, Any]):
        logging.debug(
            f"AnalysisPresenter: Received plot_added_to_dm for plot ID {plot_id}"
        )
        if self.analysis_view_widget:
            # Ensure data_manager is available to pass to create_interactive_chart_card
            if not self.data_manager:
                logging.error(
                    "AnalysisPresenter: DataManager not available when plot added. Cannot create card."
                )
                return

            icw_instance = self.analysis_view_widget.create_interactive_chart_card(
                plot_id,
                plot_config_obj,  # Pass the plot_config_obj directly
                self.data_manager,  # For data fetching by the chart
                self,  # Pass self (AnalysisPresenter) for ICW to connect its signals
            )
            if icw_instance:
                self.interactive_chart_widgets[plot_id] = icw_instance
            else:
                logging.error(
                    f"Failed to create/get InteractiveChartWidget for plot {plot_id}"
                )
        else:
            logging.error(
                "AnalysisPresenter: AnalysisViewWidget not available to add plot card."
            )

    @Slot(str)  # plot_id
    def on_plot_deleted_from_dm(self, plot_id: str):
        logging.debug(
            f"AnalysisPresenter: Received plot_deleted_from_dm for plot ID {plot_id}"
        )
        if self.analysis_view_widget:
            self.analysis_view_widget.remove_analysis_card(plot_id)
        if plot_id in self.interactive_chart_widgets:
            del self.interactive_chart_widgets[plot_id]

    @Slot(str, object)  # plot_id, gate_obj (BaseGate derivative)
    def on_gate_added_to_dm(self, plot_id: str, gate_obj: Any):
        logging.debug(
            f"AnalysisPresenter: Received gate_added_to_dm for plot {plot_id}, gate {gate_obj.id}"
        )
        icw = self.interactive_chart_widgets.get(plot_id)
        if icw and hasattr(icw, "add_or_update_gate_in_ui"):
            icw.add_or_update_gate_in_ui(gate_obj)
        elif icw:
            logging.warning(
                f"InteractiveChartWidget for plot {plot_id} missing add_or_update_gate_in_ui method."
            )
        else:
            logging.warning(
                f"No InteractiveChartWidget found for plot {plot_id} to add gate {gate_obj.id}"
            )

    @Slot(str, object)  # plot_id, gate_obj
    def on_gate_updated_in_dm(self, plot_id: str, gate_obj: Any):
        logging.debug(
            f"AnalysisPresenter: Received gate_updated_in_dm for plot {plot_id}, gate {gate_obj.id}"
        )
        icw = self.interactive_chart_widgets.get(plot_id)
        if icw and hasattr(icw, "add_or_update_gate_in_ui"):
            icw.add_or_update_gate_in_ui(gate_obj)  # Same method for add/update
        elif icw:
            logging.warning(
                f"InteractiveChartWidget for plot {plot_id} missing add_or_update_gate_in_ui method."
            )
        else:
            logging.warning(
                f"No InteractiveChartWidget found for plot {plot_id} to update gate {gate_obj.id}"
            )

    @Slot(str, str)  # plot_id, gate_id
    def on_gate_deleted_from_dm(self, plot_id: str, gate_id: str):
        logging.debug(
            f"AnalysisPresenter: Received gate_deleted_from_dm for plot {plot_id}, gate {gate_id}"
        )
        icw = self.interactive_chart_widgets.get(plot_id)
        if icw and hasattr(icw, "remove_gate_from_ui"):
            icw.remove_gate_from_ui(gate_id)
        elif icw:
            logging.warning(
                f"InteractiveChartWidget for plot {plot_id} missing remove_gate_from_ui method."
            )
        else:
            logging.warning(
                f"No InteractiveChartWidget found for plot {plot_id} to delete gate {gate_id}"
            )

    @Slot(list)  # List of all gate objects from DataManager
    def on_all_gate_populations_recalculated_from_dm(self, all_gates: List[Any]):
        logging.debug(
            f"AnalysisPresenter: Received all_gate_populations_recalculated with {len(all_gates)} gates."
        )
        # Update all relevant InteractiveChartWidgets
        for gate_obj in all_gates:
            icw = self.interactive_chart_widgets.get(gate_obj.plot_id)
            if icw and hasattr(icw, "add_or_update_gate_in_ui"):
                icw.add_or_update_gate_in_ui(
                    gate_obj
                )  # This will refresh its table entry / chart drawing
            elif icw:
                logging.warning(
                    f"InteractiveChartWidget for plot {gate_obj.plot_id} missing add_or_update_gate_in_ui method during full recalc."
                )

    #  Slots for InteractiveChartWidget Signals
    @Slot(str, object)  # plot_id, gate_obj
    def handle_new_gate_defined_on_plot(self, plot_id: str, gate_obj: Any):
        """Called by InteractiveChartWidget when a user finishes drawing a new gate."""
        if not self.data_manager_interface:
            return
        logging.info(
            f"AnalysisPresenter: New gate defined on plot {plot_id}. Requesting DM to create: {gate_obj.name}"
        )
        self.data_manager_interface.create_gate_request(plot_id, gate_obj)

    @Slot(object)  # gate_obj
    def handle_gate_update_requested_from_plot(self, gate_obj: Any):
        """Called by InteractiveChartWidget if a gate's properties (name, color) are edited in its table."""
        if not self.data_manager_interface:
            return
        logging.info(
            f"AnalysisPresenter: Gate update requested from plot {gate_obj.plot_id}. Gate: {gate_obj.name}"
        )
        self.data_manager_interface.update_gate_request(gate_obj)

    @Slot(str)  # gate_id
    def handle_gate_delete_requested_from_plot_table(self, gate_id: str):
        """Called by InteractiveChartWidget when user requests to delete a gate from its table."""
        if not self.data_manager_interface:
            return
        logging.info(
            f"AnalysisPresenter: Gate delete requested from plot table. Gate ID: {gate_id}"
        )
        self.data_manager_interface.delete_gate_request(gate_id)

    @Slot(str)  # plot_id from AnalysisCard context menu or delete button
    def handle_delete_plot_card_request(self, plot_id: str):
        """Handles request to delete an entire plot card."""
        if not self.data_manager_interface:
            return
        reply = QMessageBox.question(
            self.analysis_view_widget.window(),
            "Confirm Delete Plot",
            f"Are you sure you want to delete this plot and all its gates?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            logging.info(f"AnalysisPresenter: Requesting deletion of plot: {plot_id}")
            self.data_manager_interface.delete_plot_request(plot_id)

    def clear(self) -> None:
        logging.info("Clearing AnalysisPresenter state.")

        if self.analysis_view_widget and self.analysis_view_widget.controlPanel:
            try:
                self.analysis_view_widget.controlPanel.create_chart_requested.disconnect(
                    self.handle_create_chart_request_from_ui
                )
            except (TypeError, RuntimeError) as e:
                logging.warning(
                    f"AnalysisPresenter: Error disconnecting controlPanel signal: {e}"
                )

        #  Crucial: Clear the view itself
        if self.analysis_view_widget:
            if hasattr(self.analysis_view_widget, "clear_all_plot_cards"):
                logging.debug(
                    "AnalysisPresenter: Requesting AnalysisViewWidget to clear all plot cards."
                )
                self.analysis_view_widget.clear_all_plot_cards()
            else:
                logging.warning(
                    "AnalysisPresenter: AnalysisViewWidget does not have clear_all_plot_cards method."
                )
            self.analysis_view_widget = None

        self.interactive_chart_widgets.clear()
        logging.debug(
            f"AnalysisPresenter: Cleared internal interactive_chart_widgets dictionary (count: {len(self.interactive_chart_widgets)})."
        )

        # Clear references to DataManager and its interface
        self.data_manager_interface = None
        self.data_manager = None
        self.segmentation_model = None  # If held

        logging.info("AnalysisPresenter cleared successfully.")
