from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway

# ------------- math helpers (small, explicit, guarded) ------------- #


def _cramers_v(table: pd.DataFrame) -> float:
    """
    Cramér’s V for association between two categorical variables.
    Inputs: contingency table (rows x cols).
    Safe-guards:
      - empty table → NaN
      - min dimension < 2 → NaN
    """
    if table.size == 0 or min(table.shape) < 2:
        return np.nan
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = table.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else np.nan


def _eta_squared_from_anova(groups: List[np.ndarray]) -> float:
    """
    Eta-squared (η²) from one-way ANOVA across groups.
    Useful as a continuous analogue to Cramér’s V (effect size).
    """
    # Need at least two non-empty groups
    valid = [g for g in groups if len(g) > 0]
    if len(valid) < 2:
        return np.nan

    f_stat, _ = f_oneway(*valid)
    k = len(valid)
    n_total = sum(len(g) for g in valid)
    df_between = k - 1
    df_within = n_total - k
    if not np.isfinite(f_stat) or df_within <= 0:
        return np.nan
    return float((df_between * f_stat) / (df_between * f_stat + df_within))


# ------------- Representation detector ------------- #


@dataclass
class RepresentationResult:
    attribute: str
    counts: Dict[str, int]
    proportions: Dict[str, float]
    benchmark: Optional[Dict[str, float]]
    chi2_pvalue: Optional[float]
    flagged: bool


class RepresentationBiasDetector:
    """
    Compare observed group distribution for a sensitive attribute against an optional benchmark.
    - If a benchmark is provided (proportions per group), perform a chi-square goodness-of-fit
      by comparing observed counts with expected=benchmark*total.
    - Flag if p < alpha.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def run(
        self, df: pd.DataFrame, attribute: str, benchmark: Optional[Dict[str, float]] = None
    ) -> RepresentationResult:
        s = df[attribute].dropna().astype("category")
        counts = s.value_counts().to_dict()
        total = max(1, int(s.shape[0]))
        proportions = {str(k): float(v / total) for k, v in counts.items()}

        chi2_p = None
        flagged = False

        if benchmark:
            # unify categories across observed + benchmark
            cats = sorted(set(list(counts.keys()) + list(benchmark.keys())))
            obs = np.array([counts.get(c, 0) for c in cats], dtype=float)
            exp = np.array([benchmark.get(c, 0.0) * total for c in cats], dtype=float)

            mask = exp > 0  # avoid zero expected counts
            if mask.sum() >= 2:
                # build 2xN table (observed vs expected) for chi-square
                chi2, p, _, _ = chi2_contingency(np.vstack([obs[mask], exp[mask]]))
                chi2_p = float(p)
                flagged = bool(p < self.alpha)

        return RepresentationResult(
            attribute=attribute,
            counts={str(k): int(v) for k, v in counts.items()},
            proportions=proportions,
            benchmark=benchmark,
            chi2_pvalue=chi2_p,
            flagged=flagged,
        )


# ------------- Statistical disparity detector ------------- #


@dataclass
class DisparityResult:
    feature: str
    attribute: str
    test: str
    pvalue: float
    flagged: bool


class StatisticalDisparityDetector:
    """
    For each feature vs a sensitive attribute:
      - Categorical feature → chi-square test of independence (feature x attribute)
      - Numeric feature → one-way ANOVA across groups of the attribute
    Flag when p < alpha.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def _is_categorical(self, s: pd.Series) -> bool:
        if (
            pd.api.types.is_categorical_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_bool_dtype(s)
        ):
            return True
        # treat small-cardinality ints as categorical (e.g., codes)
        return pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 20

    def run(
        self, df: pd.DataFrame, attribute: str, features: Optional[List[str]] = None
    ) -> List[DisparityResult]:
        results: List[DisparityResult] = []
        if features is None:
            features = [c for c in df.columns if c != attribute]

        for feat in features:
            if feat == attribute or feat not in df.columns:
                continue
            aligned = df[[attribute, feat]].dropna()
            if aligned.empty:
                continue

            y = aligned[feat]
            g = aligned[attribute].astype("category")

            if self._is_categorical(y):
                table = pd.crosstab(y, g)
                if table.size == 0 or min(table.shape) < 2:
                    continue
                _, p, _, _ = chi2_contingency(table, correction=False)
                results.append(
                    DisparityResult(feat, attribute, "chi2", float(p), bool(p < self.alpha))
                )
            else:
                # numeric feature → one-way ANOVA across attribute groups
                groups = [y[g == cat].to_numpy() for cat in g.cat.categories]
                if sum(len(arr) > 0 for arr in groups) < 2:
                    continue
                _, p = f_oneway(*groups)
                results.append(
                    DisparityResult(feat, attribute, "anova", float(p), bool(p < self.alpha))
                )

        return results


# ------------- Proxy variable detector ------------- #


@dataclass
class ProxyResult:
    feature: str
    attribute: str
    measure: str  # "cramers_v" | "eta_squared"
    strength: float
    flagged: bool


class ProxyVariableDetector:
    """
    Association of non-sensitive features with a sensitive attribute:
      - Categorical feature → Cramér’s V
      - Numeric feature → eta-squared (η²)
    Flag when association >= threshold.
    """

    def __init__(self, threshold: float = 0.30):
        self.threshold = threshold

    def _is_categorical(self, s: pd.Series) -> bool:
        if (
            pd.api.types.is_categorical_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_bool_dtype(s)
        ):
            return True
        return pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 20

    def run(
        self, df: pd.DataFrame, attribute: str, features: Optional[List[str]] = None
    ) -> List[ProxyResult]:
        results: List[ProxyResult] = []
        if features is None:
            features = [c for c in df.columns if c != attribute]

        for feat in features:
            if feat == attribute or feat not in df.columns:
                continue
            aligned = df[[attribute, feat]].dropna()
            if aligned.empty:
                continue

            y = aligned[feat]
            g = aligned[attribute].astype("category")

            if self._is_categorical(y):
                table = pd.crosstab(y, g)
                if table.size == 0 or min(table.shape) < 2:
                    continue
                v = _cramers_v(table)
                results.append(
                    ProxyResult(feat, attribute, "cramers_v", float(v), bool(v >= self.threshold))
                )
            else:
                groups = [y[g == cat].to_numpy() for cat in g.cat.categories]
                if sum(len(arr) > 0 for arr in groups) < 2:
                    continue
                eta2 = _eta_squared_from_anova(groups)
                results.append(
                    ProxyResult(
                        feat, attribute, "eta_squared", float(eta2), bool(eta2 >= self.threshold)
                    )
                )

        return results
