import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class HistogramParameters:
    x_variable: str
    num_bins: int  # This should be int
    show_mean: bool
    relative_frequency: bool

    def get_data(self, data_manager):
        """Retrieves a list of (value, sample_id) tuples for the histogram."""
        data_for_histogram = []  # List of (value, sample_id)
        if not data_manager.samples:
            return data_for_histogram

        for sample_id, sample in data_manager.samples.items():
            if not sample.is_active:
                continue

            if sample.mask_id and sample.mask_id in data_manager.masks:
                mask_attributes = data_manager.masks[sample.mask_id].attributes
                if self.x_variable in mask_attributes:
                    attr_value = mask_attributes[self.x_variable]
                    data_for_histogram.append((float(attr_value), sample_id))

        return data_for_histogram  # Returns List[Tuple[float, str]]


@dataclass
class ScatterParameters:
    x_variable: str
    y_variable: str
    color_variable: Optional[str]
    # size_variable: Optional[str] = None # UNCOMMENT and add to config widget if needed
    trendline: Optional[str]
    marginal_x: Optional[str]  # Should be bool if it's just on/off
    marginal_y: Optional[str]  # Should be bool

    def get_data(self, data_manager):
        """Retrieve data for scatter plot as a list of tuples."""
        # Each tuple: (x_val, y_val, size_val, group_val_for_color, sample_id)
        data_for_scatter = []

        if not data_manager.samples:
            logging.warning("ScatterParameters.get_data: No samples in DataManager.")
            return data_for_scatter

        for sample_id, sample in data_manager.samples.items():
            if not sample.is_active:
                continue

            if sample.mask_id and sample.mask_id in data_manager.masks:
                mask_attributes = data_manager.masks[sample.mask_id].attributes

                x_val_attr, y_val_attr = None, None

                if self.x_variable in mask_attributes:
                    x_val_attr = mask_attributes[self.x_variable]
                else:
                    logging.debug(
                        f"Scatter: Missing X attribute '{self.x_variable}' for {sample.id}. Skipping sample."
                    )
                    continue

                if self.y_variable in mask_attributes:
                    y_val_attr = mask_attributes[self.y_variable]
                else:
                    logging.debug(
                        f"Scatter: Missing Y attribute '{self.y_variable}' for {sample.id}. Skipping sample."
                    )
                    continue

                try:
                    x_val_float = float(x_val_attr)
                    y_val_float = float(y_val_attr)
                except (ValueError, TypeError):
                    logging.debug(
                        f"Scatter: Non-numeric X or Y for {sample.id} (X: {x_val_attr}, Y: {y_val_attr}). Skipping sample."
                    )
                    continue

                current_size = 1.0  # Default uniform size
                current_group_for_color = (
                    None  # Default (chart uses its own default color)
                )
                if self.color_variable == "Class":
                    if sample.class_id and sample.class_id in data_manager.classes:
                        current_group_for_color = data_manager.classes[
                            sample.class_id
                        ].name
                    else:
                        current_group_for_color = "Uncategorized"
                elif self.color_variable == "Cluster":
                    assigned_cluster_id = (
                        next(iter(sample.cluster_ids), None)
                        if sample.cluster_ids
                        else None
                    )
                    if (
                        assigned_cluster_id
                        and assigned_cluster_id in data_manager.clusters
                    ):
                        current_group_for_color = data_manager.clusters[
                            assigned_cluster_id
                        ].id[:8]
                    else:
                        current_group_for_color = "No Cluster"

                data_for_scatter.append(
                    (
                        x_val_float,
                        y_val_float,
                        current_size,
                        current_group_for_color,
                        sample_id,
                    )
                )

        return data_for_scatter  # Returns List[Tuple[float, float, float, Optional[str], str]]
