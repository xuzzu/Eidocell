from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from db.session import Base, get_db
from main import app


def pytest_addoption(parser):
    parser.addoption(
        "--real-images",
        default=None,
        help="Path to directory of real test images (e.g. blood-test/elongated)",
    )


@pytest.fixture()
def real_images_dir(request):
    """Return the real images directory, or skip if not provided/missing."""
    path = request.config.getoption("--real-images")
    if not path or not Path(path).is_dir():
        pytest.skip("--real-images not provided or directory not found")
    return str(path)


@pytest.fixture()
def db_session(tmp_path):
    """Create a fresh in-memory database for each test."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def client(tmp_path):
    """TestClient with a fresh database per test."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)

    def _override_get_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture()
def sample_images_dir(tmp_path):
    """Create a temporary directory with fake image files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        (img_dir / f"cell_{i:03d}.png").write_bytes(b"\x89PNG fake image data")
    return img_dir
