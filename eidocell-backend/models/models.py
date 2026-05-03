import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Boolean, Integer, Float, ForeignKey, Table, Column, DateTime, JSON
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
    images_directory: Mapped[str] = mapped_column(String, nullable=False)
    session_folder: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    last_opened_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    scale_factor: Mapped[float] = mapped_column(Float, default=1.0)
    scale_units: Mapped[str] = mapped_column(String, default="px")

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
    path: Mapped[str] = mapped_column(String, nullable=False)
    storage_index: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    class_id: Mapped[str | None] = mapped_column(ForeignKey("classes.id"), nullable=True, index=True)

    session: Mapped["Session"] = relationship(back_populates="samples")
    sample_class: Mapped["SampleClass | None"] = relationship(back_populates="samples")
    mask: Mapped["Mask | None"] = relationship(
        back_populates="sample", uselist=False, cascade="all, delete-orphan"
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

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    sample_id: Mapped[str] = mapped_column(
        ForeignKey("samples.id"), nullable=False, unique=True, index=True
    )
    masked_image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    attributes: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    sample: Mapped["Sample"] = relationship(back_populates="mask")


class Plot(Base):
    __tablename__ = "plots"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    chart_type: Mapped[str] = mapped_column(String, nullable=False)  # "histogram" or "scatter"
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped["Session"] = relationship()
    gates: Mapped[list["Gate"]] = relationship(
        back_populates="plot", cascade="all, delete-orphan"
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
    plot_id: Mapped[str] = mapped_column(ForeignKey("plots.id"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    gate_type: Mapped[str] = mapped_column(String, nullable=False)  # "rectangular", "polygon", "interval", "ellipse", "quadrant"
    definition: Mapped[dict] = mapped_column(JSON, nullable=False)  # coordinates/vertices/range
    color: Mapped[str] = mapped_column(String, default="#FF0000")
    parameters: Mapped[list] = mapped_column(JSON, nullable=False)  # axis names
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    parent_gate_id: Mapped[str | None] = mapped_column(
        ForeignKey("gates.id"), nullable=True, index=True
    )

    plot: Mapped["Plot"] = relationship(back_populates="gates")
    session: Mapped["Session"] = relationship()
    parent_gate: Mapped["Gate | None"] = relationship(
        remote_side="Gate.id", foreign_keys=[parent_gate_id]
    )
