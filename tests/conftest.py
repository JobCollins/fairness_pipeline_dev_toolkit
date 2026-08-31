import os

import pytest

from fairness_pipeline_dev_toolkit.llm_evals.client import FAIRPIPE_LLM_ALLOW_LIVE

# Live HTTP is forbidden by default in the client (no env required). Default
# pytest still *clears* ALLOW_LIVE so a developer shell export cannot leak into
# replay tests. @pytest.mark.live_llm opts in.


@pytest.fixture(autouse=True)
def _llm_live_call_kill_switch(request):
    """Opt in only for ``live_llm`` populate tests; forbid everywhere else."""
    if request.node.get_closest_marker("live_llm"):
        previous = os.environ.get(FAIRPIPE_LLM_ALLOW_LIVE)
        os.environ[FAIRPIPE_LLM_ALLOW_LIVE] = "1"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(FAIRPIPE_LLM_ALLOW_LIVE, None)
            else:
                os.environ[FAIRPIPE_LLM_ALLOW_LIVE] = previous
    else:
        os.environ.pop(FAIRPIPE_LLM_ALLOW_LIVE, None)
        yield


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
