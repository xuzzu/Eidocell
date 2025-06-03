# src/backend/objects/session.py
import os
from typing import Any, Dict, List, Optional


class Session:
    def __init__(
        self,
        id: str,
        name: str,
        creation_date: str,
        last_opening_date: str,
        images_directory: str,
        session_folder: str,
        processor_model_name: str,
        source_type: str = "images",
        preprocessing_script: Optional[str] = None,
        preprocessing_args: Optional[str] = None,
        masks_directory: Optional[str] = None,
        metadata_directory: Optional[str] = None,
        features_directory: Optional[str] = None,
        masked_images_directory: Optional[str] = None,
        scale_factor: Optional[float] = None,  # New
        scale_units: Optional[str] = None,  # New
    ):
        self.id: str = id
        self.name: str = name
        self.creation_date: str = creation_date
        self.last_opening_date: str = last_opening_date

        self.images_directory: str = os.path.abspath(images_directory)
        self.session_folder: str = os.path.abspath(session_folder)

        self.processor_model_name: str = processor_model_name

        self.source_type: str = source_type
        self.preprocessing_script: Optional[str] = (
            os.path.abspath(preprocessing_script) if preprocessing_script else None
        )
        self.preprocessing_args: Optional[str] = preprocessing_args

        _masks_dir_name = "masks"
        _metadata_dir_name = "metadata"
        _features_dir_name = "features"
        _masked_images_dir_name = "masked_images"

        self.masks_directory: str = (
            os.path.abspath(masks_directory)
            if masks_directory
            else os.path.join(self.session_folder, _masks_dir_name)
        )
        self.metadata_directory: str = (
            os.path.abspath(metadata_directory)
            if metadata_directory
            else os.path.join(self.session_folder, _metadata_dir_name)
        )
        self.features_directory: str = (
            os.path.abspath(features_directory)
            if features_directory
            else os.path.join(self.session_folder, _features_dir_name)
        )
        self.masked_images_directory: str = (
            os.path.abspath(masked_images_directory)
            if masked_images_directory
            else os.path.join(self.session_folder, _masked_images_dir_name)
        )

        self.scale_factor: Optional[float] = scale_factor  # New
        self.scale_units: Optional[str] = scale_units  # New

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "creation_date": self.creation_date,
            "last_opening_date": self.last_opening_date,
            "images_directory": self.images_directory,
            "session_folder": self.session_folder,
            "processor_model_name": self.processor_model_name,
            "source_type": self.source_type,
            "preprocessing_script": self.preprocessing_script,
            "preprocessing_args": self.preprocessing_args,
            "masks_directory": self.masks_directory,
            "metadata_directory": self.metadata_directory,
            "features_directory": self.features_directory,
            "masked_images_directory": self.masked_images_directory,
            "scale_factor": self.scale_factor,  # New
            "scale_units": self.scale_units,  # New
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Session":
        required_keys = [
            "id",
            "name",
            "creation_date",
            "last_opening_date",
            "images_directory",
            "session_folder",
        ]
        for key in required_keys:
            if key not in data:
                raise ValueError(
                    f"Missing key '{key}' in session data dictionary for Session.from_dict."
                )

        processor_model_name = data.get(
            "processor_model_name", data.get("processor_model", "MobileNetV3")
        )

        return Session(
            id=data["id"],
            name=data["name"],
            creation_date=data["creation_date"],
            last_opening_date=data["last_opening_date"],
            images_directory=data["images_directory"],
            session_folder=data["session_folder"],
            processor_model_name=processor_model_name,
            source_type=data.get("source_type", "images"),
            preprocessing_script=data.get("preprocessing_script"),
            preprocessing_args=data.get("preprocessing_args"),
            masks_directory=data.get("masks_directory"),
            features_directory=data.get("features_directory"),
            masked_images_directory=data.get("masked_images_directory"),
            scale_factor=data.get("scale_factor"),
            scale_units=data.get("scale_units"),
        )

    @property
    def session_info_path(self) -> str:
        return os.path.join(self.session_folder, "session_info.json")

    def __repr__(self):
        return (
            f"Session(id='{self.id}', name='{self.name}', folder='{self.session_folder}', "
            f"model='{self.processor_model_name}', source_type='{self.source_type}', "
            f"scale='{self.scale_factor} {self.scale_units if self.scale_units else ''}')"
        )  # Updated repr
