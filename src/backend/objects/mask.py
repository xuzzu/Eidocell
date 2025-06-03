# backend/objects/mask.py
from typing import Any, Dict, Optional

import numpy as np  # For type hinting mask_data


class Mask:
    def __init__(
        self,
        id: str,
        image_id: str,
        attributes: Optional[Dict[str, Any]] = None,
        masked_image_path: Optional[str] = None,  # Path to the .png preview
        mask_data: Optional[np.ndarray] = None,
    ):  # Actual mask numpy array
        self.id = id
        self.image_id = image_id  # ID of the Sample this mask belongs to
        self.attributes: Dict[str, Any] = attributes if attributes is not None else {}
        self.masked_image_path: Optional[str] = (
            masked_image_path  # Path to the visual preview (e.g., .png)
        )
        self.mask_data: Optional[np.ndarray] = mask_data  # The actual binary mask data

    def to_dict(self) -> dict:
        """
        Converts the Mask object to a dictionary for storage in the database (masks table).
        Note: mask_data itself is not stored here; it's in the consolidated masks.npy.
        """
        return {
            "id": self.id,
            "image_id": self.image_id,
            "attributes": self.attributes,  # Attributes are still stored in mask_attributes table
            "masked_image_path": self.masked_image_path,
        }

    @staticmethod
    def from_dict(data: dict) -> "Mask":
        """
        Creates a Mask object from a dictionary (typically from DB 'masks' table row).
        The actual mask_data (numpy array) is loaded separately by DataManager.
        """
        return Mask(
            id=data["id"],
            image_id=data["image_id"],
            attributes=data.get(
                "attributes", {}
            ),  # Attributes loaded separately by DataManager via get_mask_attributes
            masked_image_path=data.get("masked_image_path"),
            mask_data=None,  # mask_data will be populated by DataManager
        )

    def __repr__(self):
        has_data_str = "yes" if self.mask_data is not None else "no"
        return (
            f"Mask(id='{self.id}', image_id='{self.image_id}', "
            f"has_mask_data='{has_data_str}', attributes_count={len(self.attributes)})"
        )
