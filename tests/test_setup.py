"""Smoke test: confirms the project installs and the package can be imported."""

import occupancy


def test_package_imports():
    """The occupancy package should import and report a version string."""
    assert isinstance(occupancy.__version__, str)
