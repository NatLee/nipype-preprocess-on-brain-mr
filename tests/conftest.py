import os

import pytest

from phantom import make_phantom

# No network calls from nipype's version check during tests.
os.environ.setdefault("NIPYPE_NO_ET", "1")


@pytest.fixture
def phantom():
    """A small synthetic brain with known labels (see tests/phantom.py)."""
    return make_phantom()


@pytest.fixture
def phantom_files(phantom, tmp_path):
    """The phantom saved as image + label map NIfTI files."""
    return phantom.save(tmp_path)
