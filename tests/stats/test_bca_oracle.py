"""Oracle tests for BCa percentile units (Wave 1b / BL-013).

The older BCa tests only asserted finite ordered floats and passed for seven
releases while both bounds sat below the true mean. These oracles would have
caught that failure.

SciPy dependency note
---------------------
``scipy>=1.11`` is a **runtime** dependency in ``pyproject.toml`` (not merely a
dev extra). CI installs ``pip install -e ".[dev,api]"``, which pulls that floor.
``scipy.stats.bootstrap(..., method="BCa")`` for single-sample statistics has
existed since SciPy 1.7.0, so every version allowed by the declared floor
supports the oracle. These tests **fail loudly** (import / version assert /
BCa call) rather than skip if SciPy is missing or too old.
"""

from __future__ import annotations

from importlib.metadata import version as pkg_version

import numpy as np
import pytest
import scipy
from scipy.stats import bootstrap as scipy_bootstrap

from fairness_pipeline_dev_toolkit.stats.bootstrap import bootstrap_ci

# Matches pyproject.toml / requirements.in declared floor.
_MIN_SCIPY = (1, 11, 0)

# Wave 0 / Wave 1b reproduction: fairpipe with unscaled q in [0,1] on this
# (data, B, seed) produced bounds clustered in the lower tail.
_BUGGY_ARANGE100_BCA = (39.515380436448034, 43.20968728152493)

# Absolute tolerance vs SciPy on the same n_resamples/seed protocol.
# Observed gap after the units fix is ~0.01 on each bound. The collapsed-bug
# interval sits ~4.5–12 away from SciPy, so abs=0.5 rejects the bug while
# absorbing RNG/interpolation differences across SciPy minor versions.
_SCIPY_ABS_TOL = 0.5


def _scipy_version_tuple() -> tuple[int, ...]:
    parts: list[int] = []
    for p in pkg_version("scipy").split("."):
        digits = "".join(c for c in p if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _require_scipy_for_bca_oracle() -> None:
    """Hard fail if SciPy cannot support the oracle (never skip)."""
    ver = _scipy_version_tuple()
    assert ver >= _MIN_SCIPY, (
        f"BCa oracle requires scipy>={'.'.join(map(str, _MIN_SCIPY))} "
        f"(fairpipe declared floor); got {scipy.__version__}. "
        "Install the package deps (CI: pip install -e '.[dev,api]'), do not skip."
    )


class TestBCaOracle:
    def test_bca_agrees_with_scipy_arange100(self):
        _require_scipy_for_bca_oracle()
        data = np.arange(100, dtype=float)
        B = 2000
        fp_lo, fp_hi = bootstrap_ci(data, np.mean, B=B, level=0.95, method="bca", random_state=42)
        rng = np.random.default_rng(42)
        sp = scipy_bootstrap(
            (data,),
            np.mean,
            n_resamples=B,
            confidence_level=0.95,
            method="BCa",
            random_state=rng,
        )
        sp_lo = float(sp.confidence_interval.low)
        sp_hi = float(sp.confidence_interval.high)

        assert fp_lo == pytest.approx(sp_lo, abs=_SCIPY_ABS_TOL)
        assert fp_hi == pytest.approx(sp_hi, abs=_SCIPY_ABS_TOL)

    def test_bca_interval_straddles_mean_on_symmetric_sample(self):
        """Cheap assertion that alone rejects the collapsed lower-tail bug."""
        data = np.arange(100, dtype=float)
        mean = float(np.mean(data))  # 49.5
        lo, hi = bootstrap_ci(data, np.mean, B=2000, level=0.95, method="bca", random_state=42)
        assert lo < mean < hi, f"BCa [{lo}, {hi}] does not contain mean {mean}"

    def test_bca_is_not_the_unscaled_percentile_bug(self):
        """Regression guard: a silent revert to q-in-[0,1] is caught loudly."""
        data = np.arange(100, dtype=float)
        lo, hi = bootstrap_ci(data, np.mean, B=2000, level=0.95, method="bca", random_state=42)
        bug_lo, bug_hi = _BUGGY_ARANGE100_BCA
        # Unscaled bug produced both bounds within ~0.01 of this pair.
        same_as_bug = abs(lo - bug_lo) < 0.5 and abs(hi - bug_hi) < 0.5
        assert not same_as_bug, (
            f"BCa [{lo}, {hi}] matches the pre-fix collapsed interval "
            f"[{bug_lo}, {bug_hi}] — percentile units likely reverted"
        )
        # Extra belt: corrected lower bound sits near SciPy (~44), not ~39.5.
        assert lo > 43.0
