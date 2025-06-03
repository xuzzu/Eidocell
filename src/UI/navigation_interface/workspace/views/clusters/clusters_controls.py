# UI/navigation_interface/workspace/views/clusters/clusters_controls.py
import logging

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QGroupBox
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import SubtitleLabel
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    ComboBox,
    CompactDoubleSpinBox,
    FluentIcon,
    PrimaryPushButton,
    PushButton,
    Slider,
    ToolButton,
)

from backend.config import CLUSTERING_DEFAULT_N_CLUSTERS, DEFAULT_MODELS_DICT
from UI.common.style_sheet import EidocellStyleSheet


#  Parameter Widgets for Dimensionality Reduction (Keep as is from previous version)
class BaseDimRedParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.params_layout = QVBoxLayout(self)
        self.params_layout.setContentsMargins(
            5, 5, 5, 5
        )  # Reduced margins for tighter look
        self.params_layout.setSpacing(6)  # Reduced spacing

    def get_params(self) -> dict:
        raise NotImplementedError


class PCAParamsWidget(BaseDimRedParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_components_label = BodyLabel("N Components:", self)  # Simpler label
        self.n_components_label.setObjectName("pcaNComponentsLabel")
        self.n_components_spinbox = CompactDoubleSpinBox(self)
        self.n_components_spinbox.setRange(0.01, 0.99)
        self.n_components_spinbox.setValue(0.95)
        self.n_components_spinbox.setDecimals(3)
        self.n_components_spinbox.setSingleStep(0.01)
        layout = QHBoxLayout()
        layout.addWidget(self.n_components_label)
        layout.addWidget(self.n_components_spinbox)
        layout.setAlignment(Qt.AlignTop)
        self.params_layout.addLayout(layout)

    def get_params(self) -> dict:
        return {"n_components": self.n_components_spinbox.value()}


class UMAPParamsWidget(BaseDimRedParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_neighbors_label = BodyLabel("N Neighbors:", self)
        self.n_neighbors_label.setObjectName("umapNNeighborsLabel")
        self.n_neighbors_spinbox = CompactDoubleSpinBox(self)
        self.n_neighbors_spinbox.setRange(2, 200)
        self.n_neighbors_spinbox.setValue(15)
        self.min_dist_label = BodyLabel("Min Distance:", self)
        self.min_dist_label.setObjectName("umapMinDistLabel")
        self.min_dist_spinbox = CompactDoubleSpinBox(self)
        self.min_dist_spinbox.setRange(0.0, 0.99)
        self.min_dist_spinbox.setValue(0.1)
        self.min_dist_spinbox.setDecimals(2)
        self.min_dist_spinbox.setSingleStep(0.01)
        self.n_components_umap_label = BodyLabel("N Components:", self)
        self.n_components_umap_label.setObjectName("umapNComponentsLabel")
        self.n_components_umap_spinbox = CompactDoubleSpinBox(self)
        self.n_components_umap_spinbox.setRange(2, 100)
        self.n_components_umap_spinbox.setValue(10)
        layout = QGridLayout()
        layout.addWidget(self.n_neighbors_label, 0, 0)
        layout.addWidget(self.n_neighbors_spinbox, 0, 1)
        layout.addWidget(self.min_dist_label, 1, 0)
        layout.addWidget(self.min_dist_spinbox, 1, 1)
        layout.addWidget(self.n_components_umap_label, 2, 0)
        layout.addWidget(self.n_components_umap_spinbox, 2, 1)
        self.params_layout.addLayout(layout)

    def get_params(self) -> dict:
        return {
            "n_neighbors": int(self.n_neighbors_spinbox.value()),
            "min_dist": self.min_dist_spinbox.value(),
            "n_components": int(self.n_components_umap_spinbox.value()),
        }


class NoDimRedParamsWidget(BaseDimRedParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def get_params(self) -> dict:
        return {}


#  End Parameter Widgets


class ControlPanel(QWidget):  # Changed from QFrame to QWidget, will contain a QFrame
    """Dynamic control panel for clustering parameters with consistent styling."""

    analysis_requested = Signal(dict)
    reset_analysis_requested = Signal()  # Signal for reset button

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent

        # Apply Stylesheet to the ControlPanel itself
        EidocellStyleSheet.CLUSTERS_CONTROL_PANEL.apply(self)
        # Main Frame: This will be the styled container
        self.main_styled_frame = QFrame(self)
        self.main_styled_frame.setObjectName("controlPanelStyledFrame")

        # Apply Subtle Shadow Effect to the main_styled_frame
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setXOffset(0)
        shadow.setYOffset(1)  # Slight downward offset for a more natural look
        shadow.setColor(QColor(0, 0, 0, 25))  # Adjusted alpha for subtlety
        self.main_styled_frame.setGraphicsEffect(shadow)

        # Outer layout for ControlPanel itself, to hold and center main_styled_frame
        outer_control_panel_layout = QVBoxLayout(
            self
        )  # This is the layout OF the ControlPanel widget
        outer_control_panel_layout.addWidget(self.main_styled_frame)
        outer_control_panel_layout.setContentsMargins(
            10, 10, 10, 10
        )  # Padding around the styled frame
        self.setLayout(outer_control_panel_layout)

        # Layout INSIDE the main_styled_frame
        self.vBoxLayout = QVBoxLayout(self.main_styled_frame)
        self.vBoxLayout.setContentsMargins(
            20, 20, 20, 20
        )  # Padding inside the styled frame
        self.vBoxLayout.setSpacing(15)  # Spacing between groups/elements

        #  Title
        title_label = QLabel(
            "Clustering Paremeters", self.main_styled_frame
        )  # Using SubtitleLabel
        title_label.setObjectName("panelTitleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        self.vBoxLayout.addWidget(title_label)

        #  1. Feature Sources
        self.feature_sources_group = QGroupBox(
            "Feature Sources", self.main_styled_frame
        )
        fs_layout = QVBoxLayout(self.feature_sources_group)
        fs_layout.setSpacing(8)
        fs_layout.setContentsMargins(0, 10, 0, 0)
        self.cb_use_dl_features = CheckBox(
            "Deep Learning Features", self.feature_sources_group
        )
        self.cb_use_morph_features = CheckBox(
            "Morphological Features", self.feature_sources_group
        )
        self.cb_use_meta_features = CheckBox(
            "Metadata Features", self.feature_sources_group
        )
        self.cb_use_meta_features.setEnabled(False)
        fs_layout.addWidget(self.cb_use_dl_features)
        fs_layout.addWidget(self.cb_use_morph_features)
        fs_layout.addWidget(self.cb_use_meta_features)
        self.vBoxLayout.addWidget(self.feature_sources_group)

        #  2. Feature Extractor Model (Conditional)
        self.extractor_model_group = QGroupBox(
            "DL Feature Extractor", self.main_styled_frame
        )
        em_layout = QVBoxLayout(self.extractor_model_group)
        em_layout.setSpacing(8)
        em_layout.setContentsMargins(0, 10, 0, 0)
        self.extractor_model_selector = ComboBox(self.extractor_model_group)
        self.finetune_button = PushButton(
            FluentIcon.TRAIN, "Fine-tune Extractor", self.extractor_model_group
        )
        self.finetune_button.hide()
        # em_layout.addWidget(BodyLabel("Backbone Model:", self.extractor_model_group))
        em_layout.addWidget(self.extractor_model_selector)
        em_layout.addWidget(self.finetune_button)
        self.vBoxLayout.addWidget(self.extractor_model_group)

        #  3. Dimensionality Reduction (Conditional)
        self.dim_reduction_group = QGroupBox(
            "Dimensionality Reduction", self.main_styled_frame
        )
        dr_layout = QVBoxLayout(self.dim_reduction_group)
        dr_layout.setSpacing(8)
        dr_layout.setContentsMargins(0, 10, 0, 0)
        self.dim_reduction_method_selector = ComboBox(self.dim_reduction_group)
        self.dim_reduction_method_selector.addItems(["PCA", "UMAP"])
        dr_layout.addWidget(self.dim_reduction_method_selector)

        self.dim_reduction_params_stack = QStackedWidget(self.dim_reduction_group)
        self.dim_reduction_params_stack.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self.pca_params_widget = PCAParamsWidget(self.dim_reduction_params_stack)
        self.umap_params_widget = UMAPParamsWidget(self.dim_reduction_params_stack)
        self.no_dim_red_params_widget = NoDimRedParamsWidget(
            self.dim_reduction_params_stack
        )
        self.dim_reduction_params_stack.addWidget(self.pca_params_widget)
        self.dim_reduction_params_stack.addWidget(self.umap_params_widget)
        dr_layout.addWidget(self.dim_reduction_params_stack)
        self.vBoxLayout.addWidget(self.dim_reduction_group)

        #  4. K-Means Clustering (Conditional)
        self.kmeans_group = QGroupBox("K-Means Clustering", self.main_styled_frame)
        km_layout = QVBoxLayout(self.kmeans_group)
        km_layout.setSpacing(8)
        km_layout.setContentsMargins(0, 10, 0, 0)
        self.clusters_label = BodyLabel(
            f"Clusters (K): {CLUSTERING_DEFAULT_N_CLUSTERS}", self.kmeans_group
        )
        self.clusters_label.setObjectName("clustersLabel")
        self.clusters_slider = Slider(Qt.Horizontal, self.kmeans_group)
        self.clusters_slider.setRange(2, 100)
        self.clusters_slider.setValue(CLUSTERING_DEFAULT_N_CLUSTERS)
        km_layout.addWidget(self.clusters_label)
        km_layout.addWidget(self.clusters_slider)
        self.vBoxLayout.addWidget(self.kmeans_group)

        #  Spacer & Action Buttons
        self.vBoxLayout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        self.startButton = PrimaryPushButton(
            FluentIcon.PLAY, "Start Analysis", self.main_styled_frame
        )
        self.resetButton = PushButton(
            FluentIcon.SYNC, "Reset Analysis", self.main_styled_frame
        )
        self.vBoxLayout.addWidget(self.startButton)
        self.vBoxLayout.addWidget(self.resetButton)

        self.setFixedWidth(
            320
        )  # Control panel width, slightly wider for better padding

        self._populate_extractor_models()
        self._connect_signals()
        self._update_visibility()

    def _populate_extractor_models(self):
        self.extractor_model_selector.addItems(list(DEFAULT_MODELS_DICT.keys()))
        if DEFAULT_MODELS_DICT:
            self.extractor_model_selector.setCurrentIndex(0)

    def populate_models(self, available_models: dict, current_model_name: str):
        self.extractor_model_selector.blockSignals(True)
        self.extractor_model_selector.clear()
        model_names = list(available_models.keys())
        if not model_names:
            logging.warning("No feature extractor models available for ControlPanel.")
            self.extractor_model_selector.blockSignals(False)
            return
        self.extractor_model_selector.addItems(model_names)
        if current_model_name in model_names:
            self.extractor_model_selector.setCurrentText(current_model_name)
        else:
            self.extractor_model_selector.setCurrentIndex(0)
        self.extractor_model_selector.blockSignals(False)

    def _connect_signals(self):
        self.cb_use_dl_features.stateChanged.connect(self._update_visibility)
        self.cb_use_morph_features.stateChanged.connect(self._update_visibility)
        self.cb_use_meta_features.stateChanged.connect(self._update_visibility)
        self.dim_reduction_method_selector.currentTextChanged.connect(
            self._on_dim_red_method_changed
        )
        self.clusters_slider.valueChanged.connect(
            lambda value: self.clusters_label.setText(f"Clusters (K): {value}")
        )
        self.startButton.clicked.connect(self._on_start_analysis_clicked)
        self.resetButton.clicked.connect(self.reset_analysis_requested)  # Emit signal

    @Slot()
    def _update_visibility(self):
        any_feature_selected = (
            self.cb_use_dl_features.isChecked()
            or self.cb_use_morph_features.isChecked()
            or self.cb_use_meta_features.isChecked()
        )
        self.extractor_model_group.setVisible(self.cb_use_dl_features.isChecked())
        self.dim_reduction_group.setVisible(any_feature_selected)
        self.kmeans_group.setVisible(any_feature_selected)
        self.startButton.setEnabled(any_feature_selected)
        if any_feature_selected:
            self._on_dim_red_method_changed(
                self.dim_reduction_method_selector.currentText()
            )

    @Slot(str)
    def _on_dim_red_method_changed(self, method_name: str):
        widget_map = {"PCA": self.pca_params_widget, "UMAP": self.umap_params_widget}
        self.dim_reduction_params_stack.setCurrentWidget(
            widget_map.get(method_name, self.no_dim_red_params_widget)
        )

    @Slot()
    def _on_start_analysis_clicked(self):
        #  Collect all parameters
        params = {
            "feature_sources": {
                "deep_learning": self.cb_use_dl_features.isChecked(),
                "morphological": self.cb_use_morph_features.isChecked(),
                # "metadata": self.cb_use_meta_features.isChecked() # Placeholder
            }
        }

        if params["feature_sources"]["deep_learning"]:
            params["dl_feature_extractor_settings"] = {
                "model_name": self.extractor_model_selector.currentText()
            }

        selected_dr_method = self.dim_reduction_method_selector.currentText()
        current_dr_widget = self.dim_reduction_params_stack.currentWidget()
        params["dim_reduction_settings"] = {
            "method": selected_dr_method if selected_dr_method != "None" else None,
            "params": (
                current_dr_widget.get_params()
                if isinstance(current_dr_widget, BaseDimRedParamsWidget)
                else {}
            ),
        }

        params["kmeans_settings"] = {
            "n_clusters": self.clusters_slider.value(),
            # "find_k_elbow": self.cb_find_k_elbow.isChecked() # If using elbow method
        }

        logging.debug(f"ControlPanel emitting analysis_requested with params: {params}")
        self.analysis_requested.emit(params)
