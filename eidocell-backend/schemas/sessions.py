from datetime import datetime
from pydantic import BaseModel


class SessionCreate(BaseModel):
    name: str


class SessionUpdate(BaseModel):
    name: str | None = None
    scale_factor: float | None = None
    scale_units: str | None = None
    channel_names: list[str] | None = None


class SessionOut(BaseModel):
    id: str
    name: str
    images_directory: str | None = None
    session_folder: str
    created_at: datetime
    last_opened_at: datetime
    scale_factor: float
    scale_units: str
    channel_count: int
    channel_names: list[str] | None = None
    sample_count: int

    model_config = {"from_attributes": True}


class SessionListItem(BaseModel):
    id: str
    name: str
    images_directory: str | None = None
    created_at: datetime
    last_opened_at: datetime
    sample_count: int

    model_config = {"from_attributes": True}
