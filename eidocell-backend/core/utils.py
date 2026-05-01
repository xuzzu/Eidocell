import fcntl
import logging
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

logger = logging.getLogger("eidocell")


def random_color() -> str:
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


@contextmanager
def file_lock(path: Path, *, shared: bool = False):
    """Acquire a file lock (shared for reads, exclusive for writes).

    Uses a .lock sidecar file so readers and writers coordinate.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w")
    try:
        op = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(lock_file, op)
        yield
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def npy_save(path: Path, array: np.ndarray, *, allow_pickle: bool = False) -> None:
    """Atomically save a numpy array with exclusive file locking."""
    import os
    with file_lock(path, shared=False):
        # Use a temp dir to handle np.save's auto .npy extension behaviour
        tmp_dir = tempfile.mkdtemp(dir=path.parent)
        tmp_file = Path(tmp_dir) / "data"
        try:
            np.save(str(tmp_file), array, allow_pickle=allow_pickle)
            # np.save may add .npy extension
            actual = tmp_file.with_suffix(".npy") if tmp_file.with_suffix(".npy").exists() else tmp_file
            actual.replace(path)
            logger.debug("Saved %s (shape=%s)", path.name, array.shape)
        except BaseException:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        else:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


def npy_load(path: Path, *, allow_pickle: bool = False) -> np.ndarray:
    """Load a numpy array with shared file locking."""
    with file_lock(path, shared=True):
        return np.load(path, allow_pickle=allow_pickle)


def load_session_features(session_folder: str, indices: list[int] | None = None) -> np.ndarray:
    features_path = Path(session_folder) / "features" / "session_features.npy"
    if not features_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No features extracted yet. Run feature extraction first.")
    all_features = npy_load(features_path)
    if indices is not None:
        return all_features[indices]
    return all_features


def get_active_samples(db, session_id: str):
    from models.models import Sample
    return (
        db.query(Sample)
        .filter(Sample.session_id == session_id, Sample.is_active == True)
        .order_by(Sample.storage_index)
        .all()
    )
