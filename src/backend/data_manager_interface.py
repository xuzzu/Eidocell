# backend/data_manager_interface.py
from typing import Any, Dict, List, Optional  # For type hints

from PySide6.QtCore import QObject, Signal


class DataManagerInterface(QObject):
    """
    Interface living in the main thread for presenters to interact with DataManager.
    Emits signals to request operations from the DataManager running in a worker thread.
    """

    # Session/Loading Requests
    request_load_session = Signal()
    request_load_images_from_folder = Signal(str)  # folder_path
    request_samples = Signal()  # Request for current samples dict

    # Feature Requests
    request_extract_all_features = Signal()
    request_extract_feature = Signal(
        str
    )  # image_id (Note: current DM refactor makes this behave like extract_all)
    request_delete_features = Signal(
        str
    )  # image_id (Note: current DM refactor zeros out features, marks for resave)

    # Clustering Requests
    request_perform_clustering = Signal(dict)  # n_clusters, n_iter, n_redo, find_k
    request_split_cluster = Signal(
        str, int, int, int
    )  # cluster_id, n_clusters, n_iter, n_redo
    request_merge_clusters = Signal(list)  # list of cluster_ids
    request_clear_clusters = Signal()
    request_delete_cluster = Signal(str)  # cluster_id

    # Class Requests
    request_create_class = Signal(str, str)  # name, color
    request_assign_images_to_class = Signal(list, str)  # image_ids, new_class_id
    request_delete_class = Signal(str)  # class_id
    request_rename_class = Signal(str, str)  # class_id, new_name

    # Plot and Gate Requests
    request_create_plot = Signal(str, str, str, object)
    request_delete_plot = Signal(str)  # plot_id
    request_create_gate = Signal(str, object)
    request_update_gate = Signal(object)
    request_delete_gate = Signal(str)  # gate_id

    # Mask Requests
    request_process_new_mask = Signal(str, object, object, str)
    request_delete_mask = Signal(str)  # mask_id (or image_id if deleting by image)

    # Processor Update Request
    request_update_processor = Signal(dict)  # new_settings dict

    # Export Request
    request_export_data = Signal(dict)  # params dict

    # Scale Request
    request_set_session_scale = Signal(float, str)

    # Classification
    request_train_classifier = Signal(str, int, float, int, int, dict)
    request_run_classification = Signal(str)

    # DB Flush Request (also implies persisting dirty .npy files)
    request_flush_db = Signal(object)

    def load_session(self):
        self.request_load_session.emit()

    def load_images_from_folder(self, folder_path: str):
        self.request_load_images_from_folder.emit(folder_path)

    def get_current_samples(self):
        self.request_samples.emit()

    def extract_all_features(self):
        self.request_extract_all_features.emit()

    def extract_feature(self, image_id: str):  # Behavior changed in DM
        self.request_extract_feature.emit(image_id)

    def delete_features(self, image_id: str):  # Behavior changed in DM
        self.request_delete_features.emit(image_id)

    def perform_clustering(self, ui_params: dict):
        self.request_perform_clustering.emit(ui_params)

    def split_cluster(self, cluster_id: str, n_clusters: int, n_iter: int, n_redo: int):
        self.request_split_cluster.emit(cluster_id, n_clusters, n_iter, n_redo)

    def merge_clusters(self, cluster_ids: list):
        self.request_merge_clusters.emit(cluster_ids)

    def clear_clusters(self):
        self.request_clear_clusters.emit()

    def delete_cluster(self, cluster_id: str):
        self.request_delete_cluster.emit(cluster_id)

    def create_class(self, name: str, color: str):
        self.request_create_class.emit(name, color)

    def assign_images_to_class(self, image_ids: list, new_class_id: str):
        self.request_assign_images_to_class.emit(image_ids, new_class_id)

    def delete_class(self, class_id: str):
        self.request_delete_class.emit(class_id)

    def rename_class(self, class_id: str, new_name: str):
        self.request_rename_class.emit(class_id, new_name)

    def create_plot_request(
        self, plot_id: str, name: str, chart_type_key: str, parameters_obj: object
    ):
        self.request_create_plot.emit(plot_id, name, chart_type_key, parameters_obj)

    def delete_plot_request(self, plot_id: str):
        self.request_delete_plot.emit(plot_id)

    def create_gate_request(self, plot_id: str, gate_obj: object):
        self.request_create_gate.emit(plot_id, gate_obj)

    def update_gate_request(self, gate_obj: object):
        self.request_update_gate.emit(gate_obj)

    def delete_gate_request(self, gate_id: str):
        self.request_delete_gate.emit(gate_id)

    def process_new_mask_data(
        self,
        image_id: str,
        raw_mask_numpy_array: Any,
        attributes_dict: Dict[str, Any],
        masked_preview_path: str,
    ):
        self.request_process_new_mask.emit(
            image_id, raw_mask_numpy_array, attributes_dict, masked_preview_path
        )

    def delete_mask(
        self, mask_id_or_image_id: str
    ):  # Can be mask_id or image_id (DM will differentiate)
        self.request_delete_mask.emit(mask_id_or_image_id)

    def update_processor(self, settings: dict):
        self.request_update_processor.emit(settings)

    def export_data(self, params: dict):
        self.request_export_data.emit(params)

    def set_session_scale(self, scale_factor: float, units: str):
        self.request_set_session_scale.emit(scale_factor, units)

    def train_classifier(
        self,
        feature_model_name: str,
        epochs: int,
        lr: float,
        batch_size: int,
        patience: int,
        classes_data: dict,
    ):
        self.request_train_classifier.emit(
            feature_model_name, epochs, lr, batch_size, patience, classes_data
        )

    def run_classification(self, classifier_name: str):
        self.request_run_classification.emit(classifier_name)

    def flush_db(
        self, callback: Optional[callable] = None
    ):  # Callable is passed as object
        self.request_flush_db.emit(callback)
