from pydantic import BaseModel


class ExportRequest(BaseModel):
    output_directory: str
    include_classes: bool = True
    include_clusters: bool = True
    include_masks: bool = True
    include_binary_masks: bool = False
    include_csv: bool = True


class ExportResult(BaseModel):
    output_directory: str
    classes_exported: int = 0
    clusters_exported: int = 0
    masks_exported: int = 0
    binary_masks_exported: int = 0
    csv_rows: int = 0
