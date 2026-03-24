"""
**File:** ``test_version.py``
**Region:** ``{{GITHUB_REPO}}``

Description
-----------
Smoke tests ensuring the package can be imported for coverage.
"""

from __future__ import annotations

import importlib


def test_import_package_and_version_is_string() -> None:
    """
    Verify package import works and version is exposed.

    Returns:
        None.
    """

    pkg = importlib.import_module("{{PYTHON_MODULE_NAME}}")

    assert isinstance(pkg.__version__, str)
    assert pkg.__version__ != ""
