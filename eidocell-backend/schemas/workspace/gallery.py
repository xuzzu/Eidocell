from pydantic import BaseModel


class SampleOut(BaseModel):
    id: str
    filename: str
    path: str
    is_active: bool
    class_id: str | None
    class_name: str | None
    class_color: str | None
    has_mask: bool
    mask_channels: list[int] = []
    n_channels: int = 1

    model_config = {"from_attributes": True}


class SamplePage(BaseModel):
    items: list[SampleOut]
    total: int
    offset: int
    limit: int
    session_has_any_mask: bool = False


class FilterCondition(BaseModel):
    field: str  # "filename", "class_name", or a mask attribute key
    operator: str  # "==", "!=", ">", ">=", "<", "<=", "between", "contains", "not_contains"
    value: str | float | int
    value2: str | float | int | None = None  # for "between"


class SampleListParams(BaseModel):
    offset: int = 0
    limit: int = 100
    sort_by: str = "filename"
    sort_order: str = "asc"  # "asc" or "desc"
    filters: list[FilterCondition] = []


class ClassCreate(BaseModel):
    name: str
    color: str = "#808080"


class ClassUpdate(BaseModel):
    name: str | None = None
    color: str | None = None


class ClassOut(BaseModel):
    id: str
    name: str
    color: str
    sample_count: int

    model_config = {"from_attributes": True}


class BulkClassAssignment(BaseModel):
    sample_ids: list[str]
    class_id: str | None  # None to unassign
