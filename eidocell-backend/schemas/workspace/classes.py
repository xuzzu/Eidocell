from pydantic import BaseModel


class ClassSummary(BaseModel):
    id: str
    name: str
    color: str
    sample_count: int


class ClassSamplesPage(BaseModel):
    items: list[dict]  # same shape as SampleOut
    total: int
    offset: int
    limit: int


class AttributeStatistics(BaseModel):
    name: str
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None


class ClassStatistics(BaseModel):
    id: str
    name: str
    color: str
    sample_count: int
    attributes: list[AttributeStatistics]


class AttributeDistribution(BaseModel):
    name: str
    bin_edges: list[float]
    bin_counts: list[int]
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None


class SessionDistributions(BaseModel):
    sample_count: int
    attributes: list[AttributeDistribution]
