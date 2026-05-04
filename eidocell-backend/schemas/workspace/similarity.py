from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from schemas.workspace.gallery import SampleOut


class SimilarityFilter(str, Enum):
    ALL = "all"
    UNLABELED = "unlabeled"


class SimilaritySearchRequest(BaseModel):
    reference_sample_ids: list[str] = Field(..., min_length=1)
    filter_mode: SimilarityFilter = SimilarityFilter.ALL
    feature_method: str = "mobilenetv3"
    min_similarity_pct: float = Field(0.0, ge=0.0, le=100.0)
    top_k: int | None = Field(2000, ge=1)


class SimilarityHit(BaseModel):
    sample: SampleOut
    similarity_pct: float
    bucket: int  # floor(pct / 10) * 10  -> e.g. 80 means 80–90% bucket


class SimilaritySearchResponse(BaseModel):
    reference_sample_ids: list[str]
    total_candidates: int
    returned: int
    hits: list[SimilarityHit]
