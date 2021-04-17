import pytest


@pytest.fixture
def oh():
    import oh
    oh.config._config.clear()
    return oh
