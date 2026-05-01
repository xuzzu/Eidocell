from pydantic import BaseModel, Field


class RunFeatureExtractionRequest(BaseModel):
    method: str = "morphological"


class FeatureExtractionResult(BaseModel):
    processed: int
    skipped: int
    total: int
    feature_dim: int


class FeatureExtractionMethod(BaseModel):
    id: str
    name: str
    feature_dim: int


class RunDimReductionRequest(BaseModel):
    method: str = "pca"
    n_components: int = Field(2, ge=2, le=3)
    params: dict = {}


class DimReductionResult(BaseModel):
    method: str
    n_components: int
    n_samples: int
    embeddings: list[dict]
