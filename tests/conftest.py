import os

import pytest


@pytest.fixture(autouse=True)
def set_default_profile(request):
    """
    Automatically sets FPDT_PROFILE based on the test path.
    - pipeline tests => 'pipeline'
    - training tests => 'training'
    """
    path = str(request.fspath)
    if "tests/pipeline" in path:
        os.environ["FPDT_PROFILE"] = "pipeline"
    elif "tests/training" in path:
        os.environ["FPDT_PROFILE"] = "training"
    else:
        # Default: pipeline for other modules
        os.environ["FPDT_PROFILE"] = "pipeline"
