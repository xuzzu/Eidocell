"""Cosine similarity search against a centroid of reference samples."""

import math

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.utils import load_session_features
from models.models import Mask, Sample, SampleClass, Session
from schemas.workspace.similarity import (
    SimilarityFilter,
    SimilarityHit,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
)
from services._pipeline_utils import resolve_scoped_samples


class _Scope:
    """Minimal duck-typed object for resolve_scoped_samples()."""
    def __init__(self, mode: str):
        self.mode = mode
        self.class_id = None
        self.sample_ids = None


def search(
    db: DbSession, session_id: str, req: SimilaritySearchRequest
) -> SimilaritySearchResponse:
    # 1. Validate references
    refs = (
        db.query(Sample)
        .filter(
            Sample.session_id == session_id,
            Sample.id.in_(req.reference_sample_ids),
            Sample.is_active == True,  # noqa: E712
        )
        .all()
    )
    if len(refs) != len(set(req.reference_sample_ids)):
        raise HTTPException(
            status_code=400,
            detail="One or more reference samples not found or inactive",
        )

    # 2. Resolve candidate scope (also validates session exists)
    scope_mode = "unlabeled" if req.filter_mode == SimilarityFilter.UNLABELED else "all"
    session, candidates = resolve_scoped_samples(db, session_id, _Scope(scope_mode))

    # 3. Load embeddings (raises 400 with friendly message if missing)
    features = load_session_features(session.session_folder, indices=None)

    # 4. Build reference centroid
    ref_indices = [s.storage_index for s in refs if s.storage_index < features.shape[0]]
    if not ref_indices:
        raise HTTPException(
            status_code=400,
            detail="Reference samples have no extracted features",
        )
    ref_vec = features[ref_indices].mean(axis=0)
    ref_norm = np.linalg.norm(ref_vec)
    if ref_norm == 0:
        raise HTTPException(
            status_code=400,
            detail="Reference centroid is zero vector; cannot compute similarity",
        )
    ref_unit = ref_vec / ref_norm

    # 5. Filter candidates: exclude refs, exclude any without a feature row
    ref_id_set = set(req.reference_sample_ids)
    cand_samples = [
        s for s in candidates
        if s.id not in ref_id_set and s.storage_index < features.shape[0]
    ]
    if not cand_samples:
        return SimilaritySearchResponse(
            reference_sample_ids=req.reference_sample_ids,
            total_candidates=0,
            returned=0,
            hits=[],
        )

    cand_indices = np.array([s.storage_index for s in cand_samples], dtype=np.int64)
    cand_features = features[cand_indices]
    cand_norms = np.linalg.norm(cand_features, axis=1)
    # Avoid divide-by-zero; zero-norm rows get similarity 0
    safe_norms = np.where(cand_norms > 0, cand_norms, 1.0)
    sims = (cand_features @ ref_unit) / safe_norms
    sims = np.where(cand_norms > 0, sims, 0.0)
    pcts = np.clip(sims, 0.0, 1.0) * 100.0

    # 6. Threshold + sort desc + top_k
    keep_mask = pcts >= req.min_similarity_pct
    kept_idx = np.where(keep_mask)[0]
    order = kept_idx[np.argsort(-pcts[kept_idx])]
    if req.top_k is not None:
        order = order[: req.top_k]

    top_samples = [cand_samples[i] for i in order]
    top_pcts = pcts[order]

    # 7. Batch-fetch class + mask info for the result page
    top_ids = [s.id for s in top_samples]
    class_ids = {s.class_id for s in top_samples if s.class_id}
    classes_by_id = {
        c.id: c
        for c in db.query(SampleClass).filter(SampleClass.id.in_(class_ids)).all()
    } if class_ids else {}
    mask_sample_ids = (
        {m.sample_id for m in db.query(Mask.sample_id).filter(Mask.sample_id.in_(top_ids)).all()}
        if top_ids
        else set()
    )

    hits: list[SimilarityHit] = []
    for sample, pct in zip(top_samples, top_pcts):
        cls = classes_by_id.get(sample.class_id) if sample.class_id else None
        sample_dict = {
            "id": sample.id,
            "filename": sample.filename,
            "path": sample.path,
            "storage_index": sample.storage_index,
            "is_active": sample.is_active,
            "class_id": sample.class_id,
            "class_name": cls.name if cls else None,
            "class_color": cls.color if cls else None,
            "has_mask": sample.id in mask_sample_ids,
        }
        rounded = round(float(pct), 1)
        bucket = min(int(math.floor(rounded / 10.0)) * 10, 90)
        hits.append(
            SimilarityHit(
                sample=sample_dict,
                similarity_pct=rounded,
                bucket=bucket,
            )
        )

    return SimilaritySearchResponse(
        reference_sample_ids=req.reference_sample_ids,
        total_candidates=len(cand_samples),
        returned=len(hits),
        hits=hits,
    )
