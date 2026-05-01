"""Export service: export session data to files."""

import csv
import logging
import shutil
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.utils import get_active_samples
from models.models import Session, Sample, SampleClass, Mask, Cluster, sample_clusters

logger = logging.getLogger("eidocell.export")


def export_session(
    db: DbSession,
    session_id: str,
    output_directory: str,
    include_classes: bool,
    include_clusters: bool,
    include_masks: bool,
    include_csv: bool,
) -> dict:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    out_dir = Path(output_directory)
    if not out_dir.parent.exists():
        raise HTTPException(status_code=400, detail="Parent directory does not exist")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "output_directory": str(out_dir),
        "classes_exported": 0,
        "clusters_exported": 0,
        "masks_exported": 0,
        "csv_rows": 0,
    }

    samples = get_active_samples(db, session_id)

    if include_classes:
        result["classes_exported"] = _export_classes(db, session_id, samples, out_dir)

    if include_clusters:
        result["clusters_exported"] = _export_clusters(db, session_id, samples, out_dir)

    if include_masks:
        result["masks_exported"] = _export_masks(db, samples, out_dir)

    if include_csv:
        result["csv_rows"] = _export_csv(db, session_id, samples, out_dir)

    logger.info("Export to %s: classes=%d, clusters=%d, masks=%d, csv=%d",
                out_dir, result["classes_exported"], result["clusters_exported"],
                result["masks_exported"], result["csv_rows"])
    return result


def _export_classes(
    db: DbSession, session_id: str, samples: list[Sample], out_dir: Path
) -> int:
    classes_dir = out_dir / "classes"
    classes_dir.mkdir(exist_ok=True)

    classes = db.query(SampleClass).filter(SampleClass.session_id == session_id).all()
    class_map = {c.id: c.name for c in classes}

    exported = 0
    for s in samples:
        class_name = class_map.get(s.class_id, "Uncategorized")
        class_folder = classes_dir / class_name
        class_folder.mkdir(exist_ok=True)

        src = Path(s.path)
        if src.is_file():
            shutil.copy2(src, class_folder / s.filename)
            exported += 1

    return exported


def _export_clusters(
    db: DbSession, session_id: str, samples: list[Sample], out_dir: Path
) -> int:
    clusters_dir = out_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True)

    clusters = db.query(Cluster).filter(Cluster.session_id == session_id).all()
    if not clusters:
        return 0

    # Build sample → cluster mapping
    sample_cluster_map: dict[str, list[str]] = {}
    for c in clusters:
        cluster_samples = (
            db.query(Sample.id)
            .join(sample_clusters, Sample.id == sample_clusters.c.sample_id)
            .filter(sample_clusters.c.cluster_id == c.id)
            .all()
        )
        for (sid,) in cluster_samples:
            sample_cluster_map.setdefault(sid, []).append(c.id)

    exported = 0
    for s in samples:
        cluster_ids = sample_cluster_map.get(s.id, [])
        for cid in cluster_ids:
            cluster_folder = clusters_dir / cid
            cluster_folder.mkdir(exist_ok=True)
            src = Path(s.path)
            if src.is_file():
                shutil.copy2(src, cluster_folder / s.filename)
                exported += 1

    return exported


def _export_masks(db: DbSession, samples: list[Sample], out_dir: Path) -> int:
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    sample_ids = [s.id for s in samples]
    masks = {
        m.sample_id: m.masked_image_path
        for m in db.query(Mask).filter(Mask.sample_id.in_(sample_ids)).all()
        if m.masked_image_path
    }

    exported = 0
    for s in samples:
        mask_path = masks.get(s.id)
        if mask_path:
            src = Path(mask_path)
            if src.is_file():
                shutil.copy2(src, masks_dir / f"{s.id}_mask.png")
                exported += 1

    return exported


def _export_csv(
    db: DbSession, session_id: str, samples: list[Sample], out_dir: Path
) -> int:
    # Gather class names
    classes = db.query(SampleClass).filter(SampleClass.session_id == session_id).all()
    class_map = {c.id: c.name for c in classes}

    # Gather cluster assignments
    clusters = db.query(Cluster).filter(Cluster.session_id == session_id).all()
    sample_cluster_map: dict[str, list[str]] = {}
    for c in clusters:
        rows = (
            db.query(sample_clusters.c.sample_id)
            .filter(sample_clusters.c.cluster_id == c.id)
            .all()
        )
        for (sid,) in rows:
            sample_cluster_map.setdefault(sid, []).append(c.id)

    # Gather mask attributes
    sample_ids = [s.id for s in samples]
    mask_attrs = {
        m.sample_id: m.attributes or {}
        for m in db.query(Mask).filter(Mask.sample_id.in_(sample_ids)).all()
    }

    # Determine all attribute keys
    all_keys: set[str] = set()
    for attrs in mask_attrs.values():
        all_keys.update(attrs.keys())
    attr_keys = sorted(all_keys)

    csv_path = out_dir / "parameters.csv"
    fieldnames = ["sample_id", "filename", "sample_path", "class_name", "cluster_ids"] + attr_keys

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s in samples:
            row = {
                "sample_id": s.id,
                "filename": s.filename,
                "sample_path": s.path,
                "class_name": class_map.get(s.class_id, ""),
                "cluster_ids": ";".join(sample_cluster_map.get(s.id, [])),
            }
            attrs = mask_attrs.get(s.id, {})
            for k in attr_keys:
                row[k] = attrs.get(k, "")
            writer.writerow(row)

    return len(samples)
