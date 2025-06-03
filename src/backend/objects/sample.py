# backend/objects/sample.py
from typing import Any, Optional, Set

import numpy as np


class Sample:
    def __init__(
        self,
        id: str,
        path: str,
        storage_index: Optional[int] = None,  # Index in consolidated .npy files
        is_active: bool = True,  # For soft deletes
        features: Optional[np.ndarray] = None,  # Direct features, no longer path
        class_id: Optional[str] = None,
        cluster_ids: Optional[Set[str]] = None,
        mask_id: Optional[str] = None,
    ):
        self.id = id
        self.path = path
        self.storage_index = storage_index
        self.is_active = is_active
        self.features: Optional[np.ndarray] = features  # Store the actual numpy array
        self.class_id = class_id
        self.cluster_ids: Set[str] = cluster_ids if cluster_ids is not None else set()
        self.mask_id = mask_id

    def set_features(self, features: Optional[np.ndarray]):
        self.features = features

    def set_mask_id(self, mask_id: Optional[str]):
        self.mask_id = mask_id

    def add_cluster(self, cluster):  # cluster is type Cluster
        self.cluster_ids.add(cluster.id)
        cluster.add_image(self)  # Assumes Cluster.add_image exists and takes Sample

    def remove_cluster(self, cluster):  # cluster is type Cluster
        self.cluster_ids.discard(cluster.id)
        cluster.remove_image(self)  # Assumes Cluster.remove_image exists

    def add_class(self, image_class):  # image_class is type SampleClass
        self.class_id = image_class.id
        image_class.add_image(self)  # Assumes SampleClass.add_image exists

    def remove_class(self, image_class):  # image_class is type SampleClass
        if self.class_id == image_class.id:
            self.class_id = None
            image_class.remove_image(self)  # Assumes SampleClass.remove_image exists

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "storage_index": self.storage_index,
            "is_active": self.is_active,
            # "features": self.features.tolist() if self.features is not None else [], # Don't store features array in DB dict
            "class_id": self.class_id,
            "cluster_ids": list(self.cluster_ids),
            "mask_id": self.mask_id,
            # "mask_data": self.mask_data.tolist() if self.mask_data is not None else [] # Don't store mask data array in DB dict
        }

    @staticmethod
    def from_dict(data: dict) -> "Sample":
        return Sample(
            id=data["id"],
            path=data["path"],
            storage_index=data.get("storage_index"),  # Will be loaded from DB
            is_active=data.get("is_active", True),  # Will be loaded from DB
            features=None,  # Loaded separately by DataManager
            class_id=data.get("class_id"),
            cluster_ids=set(data.get("cluster_ids", [])),
            mask_id=data.get("mask_id"),
        )

    def __repr__(self):
        return (
            f"Sample(id='{self.id}', path='{self.path}', storage_index={self.storage_index}, "
            f"is_active={self.is_active}, class_id='{self.class_id}', mask_id='{self.mask_id}', "
            f"num_features={len(self.features) if self.features is not None else 0})"
        )
