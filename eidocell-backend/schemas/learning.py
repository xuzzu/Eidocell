from datetime import datetime
from pydantic import BaseModel, Field


class TrainModelRequest(BaseModel):
    name: str
    classifier_type: str = "knn"
    feature_source: str = "raw"
    params: dict = {}


class TrainModelResult(BaseModel):
    model_id: str
    name: str
    classifier_type: str
    feature_source: str
    n_classes: int
    n_training_samples: int
    accuracy: float | None
    label_mapping: dict


class RunInferenceRequest(BaseModel):
    model_id: str


class InferenceResult(BaseModel):
    total_predicted: int
    classes_created: list[str]
    assignments: dict[str, int]


class ApplyModelRequest(BaseModel):
    target_session_id: str


class TrainedModelOut(BaseModel):
    id: str
    name: str
    classifier_type: str
    feature_source: str
    feature_dim: int
    n_classes: int
    n_training_samples: int
    accuracy: float | None
    label_mapping: dict
    session_id: str
    created_at: datetime

    model_config = {"from_attributes": True}
