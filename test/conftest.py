import pytest


@pytest.fixture
def oh():
    import oh

    oh.config.reset()
    oh.registry.clear()
    return oh
