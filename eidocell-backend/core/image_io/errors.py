"""Typed exceptions for the import / image-io path."""
from __future__ import annotations


class ImageReadError(Exception):
    """Raised when a single-image read fails. The import worker logs the path
    and reason, increments skipped_count, and moves on (skip-with-warning).
    """


class ContainerParseError(Exception):
    """Raised when a CIF/RIF container is structurally invalid (no recoverable
    objects). The import job aborts and reports failure.
    """


class CsvLinkError(Exception):
    """Raised when the user-provided CSV is missing the configured filename
    column or has no parseable rows. The import job aborts.
    """
