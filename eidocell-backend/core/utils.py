import logging
import random


logger = logging.getLogger("eidocell")


def random_color() -> str:
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def get_active_samples(db, session_id: str):
    from models.models import Sample
    return (
        db.query(Sample)
        .filter(Sample.session_id == session_id, Sample.is_active == True)
        .order_by(Sample.filename)
        .all()
    )
