# scale_calibration_dialog.py
import logging
import os

import cv2
import numpy as np
import tifffile
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
)
from qfluentwidgets import (
    BodyLabel,
    ComboBox,
    DoubleSpinBox,
    FluentTitleBar,
    FluentWindow,
    PrimaryPushButton,
    PushButton,
)
from qframelesswindow import FramelessWindow

from UI.dialogs.scale_widgets import HorizontalScaleWidget, VerticalScaleWidget


class MeasurementGraphicsView(QGraphicsView):
    """Same as before, but add a small helper to get the current zoom factor."""

    measurement_updated = Signal(float)
    measurement_finished = Signal(float)
    view_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setFrameShape(QFrame.NoFrame)

        self._is_measuring = False
        self._start_point = None
        self._measurement_line = None
        self._zoom_factor = 1.0
        self._initial_fit_done = False

    def set_pixmap(self, pixmap: QPixmap):
        self._pixmap_item.setPixmap(pixmap)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self._zoom_factor = 1.0
        self._initial_fit_done = False  # ➌ reset if a new image arrives
        QTimer.singleShot(0, self._do_initial_fit)  # ➍ defer to event-loop

    def _do_initial_fit(self):
        if self._initial_fit_done or self._pixmap_item.pixmap().isNull():
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._zoom_factor = self.transform().m11()
        self._initial_fit_done = True
        self.view_changed.emit()

    def get_pixmap_dimensions(self):
        if self._pixmap_item.pixmap().isNull():
            return None
        return self._pixmap_item.pixmap().size()

    def get_view_zoom_factor(self):
        return self._zoom_factor

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        zoom_factor_delta = (
            zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        )

        new_proposed_zoom = self.transform().m11() * zoom_factor_delta
        # Prevent zooming too far in or out
        if not (0.01 < new_proposed_zoom < 200.0):
            return

        old_pos = self.mapToScene(event.position().toPoint())
        self.scale(zoom_factor_delta, zoom_factor_delta)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

        self._zoom_factor = self.transform().m11()
        self.view_changed.emit()

        if self._measurement_line:
            pen = self._measurement_line.pen()
            pen.setWidthF(max(1.0, 2 / self._zoom_factor))  # Ensure minimum pen width
            self._measurement_line.setPen(pen)

    def mousePressEvent(self, event: QMouseEvent):
        if (
            event.button() == Qt.LeftButton
            and QGuiApplication.keyboardModifiers() == Qt.ShiftModifier
        ):
            self._is_measuring = True
            self._start_point = self.mapToScene(event.pos())

            if self._measurement_line:
                self._scene.removeItem(self._measurement_line)
                self._measurement_line = None
                self.measurement_updated.emit(0.0)

            pen = QPen(
                QColor("red"), max(1.0, 2 / self._zoom_factor)
            )  # Ensure min width
            pen.setCosmetic(True)  # Pen width is in device-independent pixels
            self._measurement_line = QGraphicsLineItem()
            self._measurement_line.setPen(pen)
            self._scene.addItem(self._measurement_line)
            event.accept()
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_measuring and self._measurement_line and self._start_point:
            end_point = self.mapToScene(event.pos())
            self._measurement_line.setLine(
                self._start_point.x(),
                self._start_point.y(),
                end_point.x(),
                end_point.y(),
            )
            length = self._measurement_line.line().length()
            self.measurement_updated.emit(length)
            event.accept()
        else:
            super().mouseMoveEvent(event)
            if (
                self.dragMode() == QGraphicsView.ScrollHandDrag
                and event.buttons() & Qt.LeftButton
            ):
                self.view_changed.emit()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._is_measuring:
            self._is_measuring = False
            final_length = 0.0
            if self._measurement_line:
                final_length = self._measurement_line.line().length()
                logging.info(f"Measurement finished. Length: {final_length:.2f} pixels")
                self.measurement_finished.emit(final_length)
            self.setDragMode(QGraphicsView.ScrollHandDrag)  # Reset drag mode
            event.accept()
        else:
            super().mouseReleaseEvent(event)
            if (
                self.dragMode() == QGraphicsView.ScrollHandDrag
            ):  # Ensure emitted after pan
                self.view_changed.emit()


class ScaleCalibrationDialog(FramelessWindow):
    """Dialog that allows a user to set a global scale, and see a dynamic scale bar that updates with zoom."""

    scale_applied = Signal(
        float, str
    )  # Emitted with final (units_per_image_px, units_name)
    finished = Signal()

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent=parent)
        self.titleBar.maxBtn.hide()
        self.titleBar.minBtn.hide()
        self.titleBar.closeBtn.hide()

        self.setWindowTitle("Set Image Scale")
        self.setMinimumSize(950, 700)

        self.image_path = image_path

        # The user’s declared "real length / image dimension" => base scale factor (units per image pixel)
        self.base_scale_factor = 1.0
        self.base_units = "µm"

        # We track the last measured line in pixels for display
        self.measured_pixel_length = 0.0

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        image_grid_layout = QGridLayout()
        self.graphics_view = MeasurementGraphicsView(self)
        self.horizontal_scale = HorizontalScaleWidget(self)
        self.vertical_scale = VerticalScaleWidget(self)
        image_grid_layout.addWidget(self.graphics_view, 0, 0)
        image_grid_layout.addWidget(self.horizontal_scale, 1, 0)
        image_grid_layout.addWidget(self.vertical_scale, 0, 1)
        image_grid_layout.setRowStretch(0, 1)
        image_grid_layout.setColumnStretch(0, 1)
        image_grid_layout.setRowStretch(1, 0)
        image_grid_layout.setColumnStretch(1, 0)

        control_frame = QFrame(self)
        control_frame.setFixedWidth(360)
        control_layout = QVBoxLayout(control_frame)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_layout.setSpacing(12)

        instruction_label = BodyLabel(
            "1. (Optionally) Shift+Left-Drag to measure in pixels.\n"
            "2. Enter the real-world dimension and its units.\n"
            "3. Apply Scale. Scale bars track zoom changes live.",
            control_frame,
        )
        instruction_label.setWordWrap(True)
        control_layout.addWidget(instruction_label)

        measurement_group = QGroupBox("Measurement", control_frame)
        measurement_layout = QVBoxLayout(measurement_group)
        self.pixel_length_label = BodyLabel("Measured: 0.00 pixels", measurement_group)
        measurement_layout.addWidget(self.pixel_length_label)
        control_layout.addWidget(measurement_group)

        scale_input_group = QGroupBox("Real-World Scale", control_frame)
        scale_input_layout = QVBoxLayout(scale_input_group)
        real_length_layout = QHBoxLayout()
        real_length_layout.addWidget(BodyLabel("Real Size:", scale_input_group))

        self.real_length_spinbox = DoubleSpinBox(scale_input_group)
        self.real_length_spinbox.setRange(1e-6, 1e9)
        self.real_length_spinbox.setDecimals(3)
        self.real_length_spinbox.setValue(100.0)
        real_length_layout.addWidget(self.real_length_spinbox)
        scale_input_layout.addLayout(real_length_layout)

        # Real length ± uncertainty
        real_length_unc_layout = QHBoxLayout()
        real_length_unc_layout.addWidget(BodyLabel("± Uncertainty:", scale_input_group))
        self.real_length_unc_spinbox = DoubleSpinBox(scale_input_group)
        self.real_length_unc_spinbox.setRange(0.0, 1e9)  # 0 means "no uncertainty"
        self.real_length_unc_spinbox.setDecimals(3)
        self.real_length_unc_spinbox.setValue(0.0)
        real_length_unc_layout.addWidget(self.real_length_unc_spinbox)
        scale_input_layout.addLayout(real_length_unc_layout)

        unit_layout = QHBoxLayout()
        unit_layout.addWidget(BodyLabel("Units:", scale_input_group))
        self.units_combobox = ComboBox(scale_input_group)
        self.units_combobox.addItems(["µm", "nm", "mm", "pixels"])
        self.units_combobox.setCurrentText("µm")
        unit_layout.addWidget(self.units_combobox)
        scale_input_layout.addLayout(unit_layout)

        control_layout.addWidget(scale_input_group)

        result_group = QGroupBox("Result", control_frame)
        result_layout = QVBoxLayout(result_group)
        self.calculated_scale_label = BodyLabel(
            "Scale: 1.00 µm/pixel (base)", result_group
        )
        result_layout.addWidget(self.calculated_scale_label)
        control_layout.addWidget(result_group)

        control_layout.addStretch(1)

        button_layout = QHBoxLayout()
        self.apply_button = PrimaryPushButton("Apply Scale", control_frame)
        self.cancel_button = PushButton("Cancel", control_frame)
        button_layout.addStretch(1)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.apply_button)
        control_layout.addLayout(button_layout)

        main_layout.addLayout(image_grid_layout, 1)
        main_layout.addWidget(control_frame)

        # Load image
        pixmap = self._load_image_to_pixmap(self.image_path)
        if pixmap.isNull():
            logging.error(f"Failed to load image: {self.image_path}")
            QMessageBox.critical(
                self, "Error", f"Could not load image:\n{self.image_path}"
            )
            QTimer.singleShot(0, self.close)  # Close dialog if image fails to load
        else:
            self.graphics_view.set_pixmap(pixmap)

        # Connect signals
        self.graphics_view.measurement_updated.connect(self.update_measurement_label)
        # If you want to do something with measurement_finished, you can connect here.

        # The crucial part: whenever the view changes (zoom/pan), we re-draw the scale bars
        self.graphics_view.view_changed.connect(self.update_visual_scales)

        # Recalculate base scale factor whenever user changes spinbox or units
        self.real_length_spinbox.valueChanged.connect(self.calibrate_scale)
        self.units_combobox.currentTextChanged.connect(self.calibrate_scale)
        self.real_length_unc_spinbox.valueChanged.connect(self.calibrate_scale)
        self.apply_button.clicked.connect(self.apply_scale)
        self.cancel_button.clicked.connect(self.close)

        # Temporary solution (might not work well)
        self.apply_button.clicked.connect(self.finished.emit)
        self.cancel_button.clicked.connect(self.finished.emit)

        # Initial calibration
        self.calibrate_scale()

    def _load_image_to_pixmap(self, image_path: str) -> QPixmap:
        """Loads an image file (including TIFF) into a QPixmap."""
        if not os.path.exists(image_path):
            logging.error(f"ScaleCalibrationDialog: Image file not found: {image_path}")
            return QPixmap()

        pixmap = QPixmap()  # Initialize to empty
        if image_path.lower().endswith((".tif", ".tiff")):
            try:
                img_data_tiff = tifffile.imread(image_path)

                if (
                    img_data_tiff.ndim > 2
                    and img_data_tiff.shape[0] > 1
                    and img_data_tiff.ndim == 3
                ):
                    if img_data_tiff.shape[0] <= 4:
                        img_data_tiff = np.moveaxis(img_data_tiff, 0, -1)
                    else:
                        img_data_tiff = img_data_tiff[0]
                elif img_data_tiff.ndim > 3 and img_data_tiff.shape[0] > 1:
                    img_data_tiff = img_data_tiff[0]

                if img_data_tiff.dtype != np.uint8:
                    if np.issubdtype(img_data_tiff.dtype, np.floating):
                        min_v, max_v = np.min(img_data_tiff), np.max(img_data_tiff)
                        if max_v > min_v:
                            img_data_tiff = (
                                (img_data_tiff - min_v) / (max_v - min_v + 1e-9)
                            ) * 255
                        else:
                            img_data_tiff = (
                                np.zeros_like(img_data_tiff)
                                if min_v == 0
                                else np.full_like(img_data_tiff, 128)
                            )
                    elif img_data_tiff.max() > 255:
                        img_data_tiff = img_data_tiff / (img_data_tiff.max() / 255.0)
                    img_data_tiff = img_data_tiff.astype(np.uint8)

                height, width = img_data_tiff.shape[:2]
                bytes_per_line = (
                    width * img_data_tiff.shape[2] if img_data_tiff.ndim == 3 else width
                )
                q_image_format = QImage.Format_Grayscale8
                if img_data_tiff.ndim == 3:
                    if img_data_tiff.shape[-1] == 3:
                        q_image_format = QImage.Format_RGB888
                    elif img_data_tiff.shape[-1] == 4:
                        q_image_format = QImage.Format_RGBA8888
                    elif img_data_tiff.shape[-1] == 1:
                        img_data_tiff = img_data_tiff.squeeze(axis=-1)
                        bytes_per_line = width
                        q_image_format = QImage.Format_Grayscale8
                elif img_data_tiff.ndim == 2:
                    q_image_format = QImage.Format_Grayscale8
                else:
                    logging.error(
                        f"ScaleCalib: Unsupported TIFF shape: {img_data_tiff.shape} for {image_path}"
                    )
                    return QPixmap()

                if not img_data_tiff.flags["C_CONTIGUOUS"]:
                    img_data_tiff = np.ascontiguousarray(img_data_tiff)
                q_img = QImage(
                    img_data_tiff.data, width, height, bytes_per_line, q_image_format
                )

                if not q_img.isNull():
                    pixmap = QPixmap.fromImage(q_img)
                else:
                    logging.error(
                        f"ScaleCalib: QImage conversion failed for TIFF {image_path}"
                    )
            except Exception as e:
                logging.error(
                    f"ScaleCalib: Error loading TIFF {image_path} for QPixmap: {e}",
                    exc_info=True,
                )
        else:
            pixmap = QPixmap(image_path)

        if pixmap.isNull():
            logging.warning(
                f"ScaleCalibrationDialog: QPixmap loaded as null for {image_path}"
            )
        return pixmap

    @Slot(float)
    def update_measurement_label(self, pixel_length: float):
        self.measured_pixel_length = pixel_length
        real_world_len = self.base_scale_factor * pixel_length

        text = f"Measured: {pixel_length:.2f} px"
        if self.base_scale_factor > 0:
            text += f"  (~{real_world_len:.2f} {self.base_units})"
        self.pixel_length_label.setText(text)

    @Slot()
    def calibrate_scale(self):
        """
        Recompute the 'base scale factor' (units per image pixel) and its uncertainty.
        If something is invalid, show an error and do NOT update.
        """
        import math

        real_length = self.real_length_spinbox.value()  # X
        real_length_unc = self.real_length_unc_spinbox.value()  # ±ΔX
        units = self.units_combobox.currentText()
        pixmap_size = self.graphics_view.get_pixmap_dimensions()

        # Basic validation checks
        if real_length <= 0:
            QMessageBox.warning(
                self, "Invalid Input", "Real length must be greater than zero."
            )
            return  # Don't update scale at all

        if real_length_unc < 0:
            QMessageBox.warning(
                self, "Invalid Input", "Uncertainty must be zero or greater."
            )
            return

        if (pixmap_size is None) or (
            pixmap_size.width() == 0 and pixmap_size.height() == 0
        ):
            QMessageBox.warning(
                self,
                "No Image Data",
                "Cannot calculate scale because the image is invalid.",
            )
            return

        width_in_px = pixmap_size.width()  # or height, or a measured feature in px
        if width_in_px <= 0:
            QMessageBox.warning(
                self,
                "Invalid Image Size",
                "Image width is zero or invalid. Cannot calibrate.",
            )
            return

        # Now it's safe to calculate the scale factor: S = X / N
        new_scale_factor = real_length / width_in_px

        # Calculate uncertainty if real_length_unc > 0
        # We assume pixel dimension is exact here -> ΔN = 0
        # Propagation formula: ΔS = S * sqrt( (ΔX / X)^2 + (ΔN / N)^2 )
        # With ΔN=0: ΔS = S * (ΔX / X)
        if real_length_unc > 0:
            scale_unc = new_scale_factor * (real_length_unc / real_length)
        else:
            scale_unc = 0.0

        # Store them in the dialog's state
        self.base_scale_factor = new_scale_factor
        self.base_scale_unc = scale_unc
        self.base_units = units

        if scale_unc > 0:
            self.calculated_scale_label.setText(
                f"Scale: {new_scale_factor:.4g} ± {scale_unc:.2g} {units}/px (base)"
            )
        else:
            self.calculated_scale_label.setText(
                f"Scale: {new_scale_factor:.4g} {units}/px (base)"
            )

        self.update_visual_scales()
        self.update_measurement_label(self.measured_pixel_length)

    @Slot()
    def update_visual_scales(self):
        """
        Called whenever the user zooms (via view_changed) or changes the base scale.
        We compute the "live" scale factor (units per *screen* pixel) and pass that
        to our scale widgets so they reflect the correct scale on screen.
        """
        base_units_per_image_px = self.base_scale_factor

        view_zoom = self.graphics_view.get_view_zoom_factor()

        displayed_units_per_screen_px = base_units_per_image_px / view_zoom

        h_widget_pixels = self.horizontal_scale.width()
        v_widget_pixels = self.vertical_scale.height()

        # If either is 0, just skip
        if (
            h_widget_pixels <= 0
            or v_widget_pixels <= 0
            or displayed_units_per_screen_px <= 0
        ):
            self.horizontal_scale.setScale(0, self.base_units, 0)
            self.vertical_scale.setScale(0, self.base_units, 0)
            return

        # Update horizontal scale
        self.horizontal_scale.setScale(
            displayed_units_per_screen_px,  # units per *screen* pixel
            self.base_units,
            h_widget_pixels,  # the domain over which to draw scale
        )

        # Update vertical scale
        self.vertical_scale.setScale(
            displayed_units_per_screen_px, self.base_units, v_widget_pixels
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_visual_scales()

    @Slot()
    def apply_scale(self):
        """
        Here we apply the base scale factor across the entire software.
        The 'zoom' is just a local display convenience, not an actual data scale change.
        """
        if self.base_scale_factor > 0:
            logging.info(
                f"Applying scale: {self.base_scale_factor:.4g} {self.base_units}/pixel"
            )
            self.scale_applied.emit(self.base_scale_factor, self.base_units)
            self.close()
        else:
            QMessageBox.warning(
                self,
                "Invalid Scale",
                "Please set a positive real length and ensure you have a valid image.",
            )
