"""Bonferroni-t intervals on template-level means (Wave 3a method C2b).

Opt-in helper for LLM demographic-swap divergence/contrast
(``ci_method="template_bonferroni_t"``). The default for those metrics is an
undefined CI: the real-data check missed decision 8 at every T (issue #63).

Each *arm* (a gated dimension, or a gated−control contrast) contributes one
mean per complete template; a Student-t interval is formed for each arm at a
Bonferroni-adjusted level, and the reported interval is ``[max L_d, max U_d]``
over arms.

``bootstrap_B`` is ignored: the method is analytic.

Real-data coverage (recorded fixture template means as a finite population,
centred, resampled; S=2000; raw C2b at T=3,4 and the package floor T≥5 for
T=5,7,10):

=======  ===========  ==========
T        mean cov     worst cov
=======  ===========  ==========
3        0.790        0.736
4        0.875        0.845
5        0.916        0.906
7        0.946        0.932
10       0.937        0.929
=======  ===========  ==========

None of these cells clear decision 8 (mean ≥ 0.95 and min ≥ 0.93).
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

#: Smallest number of complete templates at which the C2b *opt-in* is defined.
#: Below this the helper raises ``too_few_templates``. The real-data check
#: (see module docstring) did not clear decision 8 at any T, so divergence /
#: contrast default to an undefined CI; this floor only gates the opt-in.
#: Never below 3 by design.
T_MIN_TEMPLATES = 5

#: Inclusive upper end of the "small-T" band. From ``T_MIN_TEMPLATES`` through
#: this value the CI is reported with a ``ci_note``; above it, no note.
T_SMALL_NOTE_UNTIL = 10


def small_template_note(T: int) -> Optional[str]:
    """Return a ``ci_note`` for the small-T band, or ``None`` when T is ample."""
    if T < T_MIN_TEMPLATES:
        return None  # callers raise / set undefined instead
    if T <= T_SMALL_NOTE_UNTIL:
        return f"small_T:T={T} (C2b with df={T - 1}; prefer T>{T_SMALL_NOTE_UNTIL})"
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
    **Not the default** for demographic-swap metrics: real-data coverage at
    T=5 / 7 / 10 was 0.916 / 0.946 / 0.937 (worst 0.906 / 0.932 / 0.929), and
    raw C2b at T=3 / 4 was 0.790 / 0.875 (worst 0.736 / 0.845) — all below
    decision 8. Opt in via ``ci_method="template_bonferroni_t"``.

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
    lowers, uppers = [], []
    all_zero = True
    for v in arrays.values():
        mean = float(v.mean())
        sd = float(v.std(ddof=1)) if T > 1 else 0.0
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
