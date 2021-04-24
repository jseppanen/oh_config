import pytest


@pytest.fixture
def oh():
    import oh
    oh.config.clear()
    oh.registry.clear()
    return oh
