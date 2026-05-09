"""Cosine similarity search against a centroid of reference samples (LanceDB-backed)."""

import math

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.storage import features as lance_features
from models.models import Mask, Sample, SampleClass
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
    method = req.feature_method

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
    _, candidates = resolve_scoped_samples(db, session_id, _Scope(scope_mode))

    # 3. Load reference vectors and build centroid
    ref_ids, ref_matrix = lance_features.load_vectors(
        session_id, method, [s.id for s in refs]
    )
    if ref_matrix.shape[0] == 0:
        raise HTTPException(
            status_code=400,
            detail="Reference samples have no extracted features",
        )
    ref_vec = ref_matrix.mean(axis=0)
    ref_norm = float(np.linalg.norm(ref_vec))
    if ref_norm == 0:
        raise HTTPException(
            status_code=400,
            detail="Reference centroid is zero vector; cannot compute similarity",
        )

    # 4. Restrict candidate set to ones that have features for this method
    ref_id_set = set(req.reference_sample_ids)
    cand_pool = [s for s in candidates if s.id not in ref_id_set]
    if not cand_pool:
        return SimilaritySearchResponse(
            reference_sample_ids=req.reference_sample_ids,
            total_candidates=0,
            returned=0,
            hits=[],
        )

    cand_ids = [s.id for s in cand_pool]
    have_ids = lance_features.has_method(session_id, method, cand_ids)
    cand_samples = [s for s in cand_pool if s.id in have_ids]
    if not cand_samples:
        return SimilaritySearchResponse(
            reference_sample_ids=req.reference_sample_ids,
            total_candidates=0,
            returned=0,
            hits=[],
        )

    # 5. Lance cosine search restricted to candidate ids
    hits_raw = lance_features.cosine_search(
        session_id,
        method,
        ref_vec,
        k=req.top_k,
        candidate_sample_ids=[s.id for s in cand_samples],
    )

    # cosine sim in [-1, 1] → percent in [0, 100] (negative similarity → 0)
    sample_by_id = {s.id: s for s in cand_samples}
    filtered: list[tuple[Sample, float]] = []
    for sid, sim in hits_raw:
        s = sample_by_id.get(sid)
        if s is None:
            continue
        pct = max(0.0, min(1.0, sim)) * 100.0
        if pct < req.min_similarity_pct:
            continue
        filtered.append((s, pct))

    if req.top_k is not None:
        filtered = filtered[: req.top_k]

    top_samples = [s for s, _ in filtered]
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
    for sample, pct in filtered:
        cls = classes_by_id.get(sample.class_id) if sample.class_id else None
        sample_dict = {
            "id": sample.id,
            "filename": sample.filename,
            "path": sample.path,
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
