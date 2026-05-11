import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Boolean, Integer, Float, ForeignKey, Table, Column, DateTime, JSON, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.session import Base


# Junction table for many-to-many: samples <-> clusters
sample_clusters = Table(
    "sample_clusters",
    Base.metadata,
    Column("sample_id", String, ForeignKey("samples.id"), primary_key=True, index=True),
    Column("cluster_id", String, ForeignKey("clusters.id"), primary_key=True, index=True),
)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String, nullable=False)
    images_directory: Mapped[str | None] = mapped_column(String, nullable=True)
    session_folder: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    last_opened_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    scale_factor: Mapped[float] = mapped_column(Float, default=1.0)
    scale_units: Mapped[str] = mapped_column(String, default="px")
    channel_count: Mapped[int] = mapped_column(Integer, default=1)
    channel_names: Mapped[list | None] = mapped_column(JSON, nullable=True)
    selected_gate_id: Mapped[str | None] = mapped_column(
        ForeignKey("gates.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    samples: Mapped[list["Sample"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    classes: Mapped[list["SampleClass"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    clusters: Mapped[list["Cluster"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    trained_models: Mapped[list["TrainedModel"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class SampleClass(Base):
    __tablename__ = "classes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    color: Mapped[str] = mapped_column(String, default="#808080")

    session: Mapped["Session"] = relationship(back_populates="classes")
    samples: Mapped[list["Sample"]] = relationship(back_populates="sample_class")


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    # Primary raw source path (debugging/re-import). May be empty for samples
    # produced from container formats (CIF/RIF) where bytes live in Lance only.
    path: Mapped[str] = mapped_column(String, nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    class_id: Mapped[str | None] = mapped_column(ForeignKey("classes.id"), nullable=True, index=True)
    # Per-sample multi-channel + import metadata. Shape:
    #   {"order": ["BF","DAPI"], "shape": [H,W,C], "raw_paths": [...], "import_id": "..."}
    channels: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    session: Mapped["Session"] = relationship(back_populates="samples")
    sample_class: Mapped["SampleClass | None"] = relationship(back_populates="samples")
    masks: Mapped[list["Mask"]] = relationship(
        back_populates="sample", cascade="all, delete-orphan"
    )
    clusters: Mapped[list["Cluster"]] = relationship(
        secondary=sample_clusters, back_populates="samples"
    )


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    color: Mapped[str] = mapped_column(String, default="#808080")
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    feature_method: Mapped[str | None] = mapped_column(String, nullable=True)

    session: Mapped["Session"] = relationship(back_populates="clusters")
    samples: Mapped[list["Sample"]] = relationship(
        secondary=sample_clusters, back_populates="clusters"
    )


class Mask(Base):
    __tablename__ = "masks"
    __table_args__ = (UniqueConstraint("sample_id", "channel_index", name="uq_masks_sample_channel"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    sample_id: Mapped[str] = mapped_column(
        ForeignKey("samples.id"), nullable=False, index=True
    )
    channel_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    segmentation_method: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    sample: Mapped["Sample"] = relationship(back_populates="masks")


class Plot(Base):
    __tablename__ = "plots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    chart_type: Mapped[str] = mapped_column(String, nullable=False)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    parent_gate_id: Mapped[str | None] = mapped_column(
        ForeignKey("gates.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped["Session"] = relationship()
    gates: Mapped[list["Gate"]] = relationship(
        back_populates="plot", cascade="all, delete-orphan",
        foreign_keys="Gate.plot_id",
    )


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String, nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    classifier_type: Mapped[str] = mapped_column(String, nullable=False)
    feature_source: Mapped[str] = mapped_column(String, nullable=False)
    feature_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String, nullable=False)
    label_mapping: Mapped[dict] = mapped_column(JSON, nullable=False)
    n_classes: Mapped[int] = mapped_column(Integer, nullable=False)
    n_training_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped["Session"] = relationship(back_populates="trained_models")


class Gate(Base):
    __tablename__ = "gates"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    plot_id: Mapped[str | None] = mapped_column(
        ForeignKey("plots.id"), nullable=True, index=True
    )
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    gate_type: Mapped[str] = mapped_column(String, nullable=False)  # "rectangular" | "polygon" | "interval" | "ellipse" | "quadrant" | "boolean"
    definition: Mapped[dict] = mapped_column(JSON, nullable=False)
    color: Mapped[str] = mapped_column(String, default="#FF0000")
    parameters: Mapped[list] = mapped_column(JSON, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)  # deprecated, kept for back-compat
    parent_gate_id: Mapped[str | None] = mapped_column(
        ForeignKey("gates.id"), nullable=True, index=True
    )
    operator: Mapped[str | None] = mapped_column(String, nullable=True)  # "AND" | "OR" for boolean gates
    source_gate_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)  # list of two gate IDs for boolean gates

    plot: Mapped["Plot | None"] = relationship(
        back_populates="gates", foreign_keys=[plot_id]
    )
    session: Mapped["Session"] = relationship(foreign_keys=[session_id])
    parent_gate: Mapped["Gate | None"] = relationship(
        remote_side="Gate.id", foreign_keys=[parent_gate_id]
    )


class Import(Base):
    __tablename__ = "imports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    source_kind: Mapped[str] = mapped_column(String, nullable=False)  # "folder" | "cif" | "rif" | "hdf5"
    source_path: Mapped[str] = mapped_column(String, nullable=False)
    csv_path: Mapped[str | None] = mapped_column(String, nullable=True)
    csv_filename_col: Mapped[str | None] = mapped_column(String, nullable=True)
    channel_grouping: Mapped[bool] = mapped_column(Boolean, default=True)
    preprocessing_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|loading|preprocessing|completed|failed|cancelled
    task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    previews_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[list | None] = mapped_column(JSON, nullable=True)  # [{"path", "reason"}]

    session: Mapped["Session"] = relationship()
