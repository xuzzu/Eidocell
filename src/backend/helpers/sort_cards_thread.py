### backend/helpers/sort_cards_thread.py
from PySide6.QtCore import QThread, Signal

from backend.data_manager import DataManager


class SortCardsThread(QThread):
    """Thread for sorting gallery cards in the background."""

    sorted_data = Signal(list)  # Signal to emit the sorted image IDs

    def __init__(self, data_manager: DataManager, sort_parameter: str, sort_order: str):
        super().__init__()
        self.data_manager = data_manager
        self.sort_parameter = sort_parameter
        self.sort_order = sort_order

    def run(self):
        """Sorts the cards based on the specified parameter and order."""
        print(f"Sorting gallery by {self.sort_parameter} in {self.sort_order} order.")

        # First check if all images have associated masks
        missing_masks = []
        for image_id, sample in self.data_manager.samples.items():
            if sample.mask_id is None or sample.mask_id not in self.data_manager.masks:
                missing_masks.append(image_id)

        if missing_masks:
            print(f"Warning: The following images are missing masks: {missing_masks}")
            # Return empty list to signal that sorting failed
            self.sorted_data.emit([])
            return

        parameter_to_attribute = {
            "Area": "area",
            "Perimeter": "perimeter",
            "Solidity": "solidity",
            "Eccentricity": "eccentricity",
            "Aspect Ratio": "aspect_ratio",
            "Mean Intensity": "mean_intensity",
            "Convexity": "convexity",
            "Compactness": "compactness",
            "Circularity": "circularity",
            "Major Axis Length": "major_axis_length",
            "Minor Axis Length": "minor_axis_length",
            "Curl": "curl",
            "Volume": "volume",
        }

        sort_ascending = self.sort_order == "Ascending"

        try:
            image_mask_dict = {}
            for image_id in self.data_manager.samples:
                image_mask_dict[image_id] = self.data_manager.samples[image_id].mask_id

            # create dictionary of image_id and attribute value
            image_attribute_dict = {}
            for image_id, mask_id in image_mask_dict.items():
                mask = self.data_manager.get_mask(mask_id)
                attribute = mask.attributes[parameter_to_attribute[self.sort_parameter]]
                image_attribute_dict[image_id] = attribute
            # sort the
            sorted_image_ids = sorted(
                image_attribute_dict,
                key=image_attribute_dict.get,
                reverse=not sort_ascending,
            )
            self.sorted_data.emit(sorted_image_ids)
        except KeyError:
            print(
                f"Warning: Sorting parameter '{self.sort_parameter}' not found in mask attributes."
            )
        except Exception as e:
            print(f"An error occurred while sorting the gallery: {e}")
            import traceback

            traceback.print_exc()
