"""Learning service: train classifiers, run inference, manage model registry."""

import logging
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session as DbSession

from core.config import MODELS_DIR
from core.processors.learning.classifiers import get_classifier
from core.storage import features as lance_features
from core.utils import random_color
from models.models import Session, Sample, SampleClass, TrainedModel

logger = logging.getLogger("eidocell.learning")


# ── Training ──────────────────────────────────────────────────────────


def train_model(
    db: DbSession,
    session_id: str,
    name: str,
    classifier_type: str,
    feature_source: str,
    params: dict,
) -> dict:
    try:
        classifier = get_classifier(classifier_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find labeled samples (non-Uncategorized)
    uncategorized = (
        db.query(SampleClass)
        .filter(SampleClass.session_id == session_id, SampleClass.name == "Uncategorized")
        .first()
    )
    uncategorized_id = uncategorized.id if uncategorized else None

    labeled_samples = (
        db.query(Sample)
        .filter(
            Sample.session_id == session_id,
            Sample.is_active == True,
            Sample.class_id.isnot(None),
            Sample.class_id != uncategorized_id,
        )
        .order_by(Sample.filename)
        .all()
    )

    if not labeled_samples:
        raise HTTPException(status_code=400, detail="No labeled samples found")

    class_ids = list({s.class_id for s in labeled_samples})
    classes = {
        c.id: c.name
        for c in db.query(SampleClass).filter(SampleClass.id.in_(class_ids)).all()
    }
    if len(classes) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 distinct classes for training",
        )

    class_names_sorted = sorted(set(classes.values()))
    name_to_index = {name: idx for idx, name in enumerate(class_names_sorted)}
    label_mapping = {str(idx): name for name, idx in name_to_index.items()}

    found_ids, features = lance_features.load_vectors(
        session_id, feature_source, [s.id for s in labeled_samples]
    )
    if features.shape[0] == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No features available for method '{feature_source}'. "
                f"Run feature extraction first."
            ),
        )
    sample_by_id = {s.id: s for s in labeled_samples}
    aligned = [sample_by_id[sid] for sid in found_ids]
    labels = np.array([name_to_index[classes[s.class_id]] for s in aligned])

    feature_dim = features.shape[1]

    classifier.train(features, labels, **params)
    predictions = classifier.predict(features)
    accuracy = float(np.mean(predictions == labels))

    from models.models import _new_id
    model_id = _new_id()
    artifact_path = MODELS_DIR / f"{model_id}.pkl"
    classifier.save(artifact_path)

    trained_model = TrainedModel(
        id=model_id,
        name=name,
        session_id=session_id,
        classifier_type=classifier_type,
        feature_source=feature_source,
        feature_dim=feature_dim,
        artifact_path=str(artifact_path),
        label_mapping=label_mapping,
        n_classes=len(classes),
        n_training_samples=len(aligned),
        accuracy=accuracy,
    )
    db.add(trained_model)
    db.commit()

    logger.info(
        "Model trained: %s (%s), %d classes, %d samples, accuracy=%.3f",
        name, classifier_type, len(classes), len(aligned), accuracy,
    )

    return {
        "model_id": model_id,
        "name": name,
        "classifier_type": classifier_type,
        "feature_source": feature_source,
        "n_classes": len(classes),
        "n_training_samples": len(aligned),
        "accuracy": accuracy,
        "label_mapping": label_mapping,
    }


# ── Inference ─────────────────────────────────────────────────────────


def run_inference(
    db: DbSession, session_id: str, model_id: str
) -> dict:
    """Run inference on unlabeled (Uncategorized) samples."""
    trained_model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not trained_model:
        raise HTTPException(status_code=404, detail="Trained model not found")
    return _apply_model_to_session(db, trained_model, session_id)


def apply_model_cross_session(
    db: DbSession, model_id: str, target_session_id: str
) -> dict:
    """Apply a trained model to a different session."""
    trained_model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not trained_model:
        raise HTTPException(status_code=404, detail="Trained model not found")
    session = db.query(Session).filter(Session.id == target_session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Target session not found")
    return _apply_model_to_session(db, trained_model, target_session_id)


def _apply_model_to_session(
    db: DbSession, trained_model: TrainedModel, session_id: str
) -> dict:
    """Core inference logic: load model, predict unlabeled, assign classes."""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    uncategorized = (
        db.query(SampleClass)
        .filter(SampleClass.session_id == session_id, SampleClass.name == "Uncategorized")
        .first()
    )
    if not uncategorized:
        raise HTTPException(status_code=400, detail="No Uncategorized class found")

    unlabeled_samples = (
        db.query(Sample)
        .filter(
            Sample.session_id == session_id,
            Sample.is_active == True,
            Sample.class_id == uncategorized.id,
        )
        .order_by(Sample.filename)
        .all()
    )

    if not unlabeled_samples:
        return {"total_predicted": 0, "classes_created": [], "assignments": {}}

    found_ids, features = lance_features.load_vectors(
        session_id, trained_model.feature_source, [s.id for s in unlabeled_samples]
    )
    if features.shape[0] == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No features available for method '{trained_model.feature_source}'. "
                f"Run feature extraction in this session first."
            ),
        )
    if features.shape[1] != trained_model.feature_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Feature dimension mismatch: model expects {trained_model.feature_dim}, "
                f"but session has {features.shape[1]}. Use the same feature extraction method."
            ),
        )

    classifier = get_classifier(trained_model.classifier_type)
    artifact_path = Path(trained_model.artifact_path)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Model artifact file not found")
    classifier.load(artifact_path)

    predictions = classifier.predict(features)
    label_mapping = trained_model.label_mapping  # {str(index): class_name}

    classes_created: list[str] = []
    class_name_to_id: dict[str, str] = {}

    existing_classes = db.query(SampleClass).filter(
        SampleClass.session_id == session_id
    ).all()
    existing_map = {c.name: c.id for c in existing_classes}

    for _, class_name in label_mapping.items():
        if class_name in existing_map:
            class_name_to_id[class_name] = existing_map[class_name]
        else:
            new_class = SampleClass(
                session_id=session_id, name=class_name, color=random_color()
            )
            db.add(new_class)
            db.flush()
            class_name_to_id[class_name] = new_class.id
            classes_created.append(class_name)

    sample_by_id = {s.id: s for s in unlabeled_samples}
    sample_ids_by_class: dict[str, list[str]] = {}
    for sid, pred_idx in zip(found_ids, predictions):
        if sid not in sample_by_id:
            continue
        class_name = label_mapping.get(str(int(pred_idx)))
        if class_name is None:
            continue
        sample_ids_by_class.setdefault(class_name, []).append(sid)

    assignments: dict[str, int] = {}
    for class_name, sids in sample_ids_by_class.items():
        db.query(Sample).filter(Sample.id.in_(sids)).update(
            {"class_id": class_name_to_id[class_name]}, synchronize_session=False
        )
        assignments[class_name] = len(sids)

    db.commit()
    logger.info(
        "Inference complete: %d predicted, classes created: %s",
        sum(assignments.values()), classes_created,
    )

    return {
        "total_predicted": sum(assignments.values()),
        "classes_created": classes_created,
        "assignments": assignments,
    }


# ── Model registry CRUD ──────────────────────────────────────────────


def list_models(db: DbSession, session_id: str | None = None) -> list[dict]:
    query = db.query(TrainedModel)
    if session_id:
        query = query.filter(TrainedModel.session_id == session_id)
    models = query.order_by(TrainedModel.created_at.desc()).all()
    return [_model_to_out(m) for m in models]


def get_model(db: DbSession, model_id: str) -> dict:
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return _model_to_out(model)


def delete_model(db: DbSession, model_id: str) -> None:
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    artifact = Path(model.artifact_path)
    if artifact.exists():
        artifact.unlink()

    db.delete(model)
    db.commit()


def _model_to_out(model: TrainedModel) -> dict:
    return {
        "id": model.id,
        "name": model.name,
        "classifier_type": model.classifier_type,
        "feature_source": model.feature_source,
        "feature_dim": model.feature_dim,
        "n_classes": model.n_classes,
        "n_training_samples": model.n_training_samples,
        "accuracy": model.accuracy,
        "label_mapping": model.label_mapping,
        "session_id": model.session_id,
        "created_at": model.created_at,
    }
