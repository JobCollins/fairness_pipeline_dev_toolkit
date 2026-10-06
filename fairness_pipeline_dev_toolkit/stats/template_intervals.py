"""Bonferroni-t intervals on per-template means.

Opt-in helper for LLM demographic-swap divergence/contrast
(``ci_method="template_bonferroni_t"``). The default for those metrics is an
undefined CI: real-data coverage did not meet the calibration criterion
(mean coverage ≥ 0.95 and worst ≥ 0.93 for a 95% interval) at any template
count (issue #63; see ``investigations/wave3a/``).

Each *arm* (a gated dimension, or a gated−control contrast) contributes one
mean per complete template; a Student-t interval is formed for each arm at a
Bonferroni-adjusted level, and the reported interval is ``[max L_d, max U_d]``
over arms.

``bootstrap_B`` is ignored: the method is analytic.

Real-data coverage (recorded fixture template means as a finite population,
centred, resampled; S=2000; raw method at T=3,4 and the package floor T≥5 for
T=5,7,10). Numbers are generated from
``investigations/wave3a/results/tmin_c2b_summary.json`` via
``investigations/wave3a/coverage_quotes.py`` (round half-up to 3 d.p.):

=======  ===========  ==========
T        mean cov     worst cov
=======  ===========  ==========
3        0.790        0.736
4        0.875        0.845
5        0.929        0.920
7        0.944        0.934
10       0.945        0.939
=======  ===========  ==========

None of these cells meet mean ≥ 0.95 and worst ≥ 0.93.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import t as student_t

from ..exceptions import IntervalUndefinedError

__all__ = [
    "T_MIN_TEMPLATES",
    "T_SMALL_NOTE_UNTIL",
    "template_bonferroni_t_max_mean",
    "small_template_note",
]

#: Smallest number of complete templates at which the opt-in method is defined.
#: Below this the helper raises ``too_few_templates``. Real-data coverage
#: (see module docstring) did not meet the calibration criterion at any T, so
#: divergence / contrast default to an undefined CI; this floor only gates the
#: opt-in. Never below 3 by design.
T_MIN_TEMPLATES = 5

#: Inclusive upper end of the "small-T" band. From ``T_MIN_TEMPLATES`` through
#: this value the CI is reported with a ``ci_note``; above it, no note.
T_SMALL_NOTE_UNTIL = 10


def small_template_note(T: int) -> Optional[str]:
    """Return a ``ci_note`` for the small-T band, or ``None`` when T is ample."""
    if T < T_MIN_TEMPLATES:
        return None  # callers raise / set undefined instead
    if T <= T_SMALL_NOTE_UNTIL:
        return (
            f"small_T:T={T} (Bonferroni-t on per-template means, df={T - 1}; "
            f"prefer T>{T_SMALL_NOTE_UNTIL})"
        )
    return None


def template_bonferroni_t_max_mean(
    arm_values: Mapping[str, Sequence[float]],
    *,
    level: float = 0.95,
) -> Tuple[float, float]:
    """Bonferroni-t intervals on each arm, inverted to a max-mean interval.

    Parameters
    ----------
    arm_values
        Mapping arm name → length-T sequence of per-template means. Every arm
        must have the same T; values must be finite.
    level
        Family-wise confidence level. Each arm's t-interval uses
        ``1 − (1 − level) / D`` with ``D = len(arm_values)`` and ``df = T − 1``.

    Returns
    -------
    (L, U)
        ``L = max_d lower_d``, ``U = max_d upper_d``.

    Raises
    ------
    IntervalUndefinedError
        ``"too_few_templates"`` if ``T < T_MIN_TEMPLATES``;
        ``"nonfinite_template_values"`` if any value is bad.

    Notes
    -----
    **Not the default** for demographic-swap metrics: real-data coverage
    (from ``investigations/wave3a/results/tmin_c2b_summary.json`` via
    ``coverage_quotes.py``) at T=5 / 7 / 10 was 0.929 / 0.944 / 0.945
    (worst 0.920 / 0.934 / 0.939), and raw coverage at T=3 / 4 was
    0.790 / 0.875 (worst 0.736 / 0.845) — all below the calibration criterion
    (mean ≥ 0.95 and worst ≥ 0.93 for a 95% interval). Opt in via
    ``ci_method="template_bonferroni_t"``.

    If every arm has zero sample SD (identical template values), the interval
    collapses to ``[m, m]`` where ``m = max_d mean_d``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level!r}")
    if not arm_values:
        raise IntervalUndefinedError("too_few_arms", "need at least one arm")
    arrays = {k: np.asarray(v, dtype=float) for k, v in arm_values.items()}
    lengths = {v.size for v in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("every arm must have the same number of templates")
    T = next(iter(lengths))
    if T < T_MIN_TEMPLATES:
        raise IntervalUndefinedError("too_few_templates", f"T={T} < {T_MIN_TEMPLATES}")
    if any(not np.all(np.isfinite(v)) for v in arrays.values()):
        raise IntervalUndefinedError("nonfinite_template_values")

    D = len(arrays)
    a_tail = (1.0 - level) / (2.0 * D)
    crit = float(student_t.ppf(1.0 - a_tail, T - 1))
    lowers: list[float] = []
    uppers: list[float] = []
    all_zero = True
    for v in arrays.values():
        mean = float(v.mean())
        sd = float(v.std(ddof=1))
        if sd > 0.0:
            all_zero = False
            se = sd / np.sqrt(T)
            lowers.append(mean - crit * se)
            uppers.append(mean + crit * se)
        else:
            lowers.append(mean)
            uppers.append(mean)
    if all_zero:
        m = max(lowers)
        return m, m
    return float(max(lowers)), float(max(uppers))
