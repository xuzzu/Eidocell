"""Preprocessing pipeline for imported images.

Public surface:

- :mod:`ops`      — pure per-image operations (pad, resize, z-score, align...)
- :mod:`pipeline` — composes ops + records per-step metadata for reproducibility
- :mod:`stats`    — dataset-level passes (target shape from staging table)
"""
from core.preprocessing.pipeline import Pipeline, build_pipeline  # noqa: F401
