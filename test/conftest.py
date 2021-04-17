import pytest


@pytest.fixture
def oh():
    import oh
    oh.config._config.clear()
    oh.registry.clear()
    return oh
