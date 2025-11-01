from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from fairlearn.reductions import ExponentiatedGradient
from sklearn.base import BaseEstimator, ClassifierMixin, clone


@dataclass
class ReductionsWrapper(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible wrapper that trains an estimator under a fairness
    constraint using Fairlearn's ExponentiatedGradient.

    Parameters
    ----------
    base_estimator : BaseEstimator
        Any sklearn classifier (e.g., GradientBoostingClassifier, XGBClassifier).
    constraint : object
        A fairlearn.reductions constraint, e.g., DemographicParity(...) or EqualizedOdds(...).
    eps : float
        Tolerance on constraint violation used by ExponentiatedGradient.
    T : int
        Max number of iterations for ExponentiatedGradient.
    kwargs : dict
        Extra kwargs forwarded to ExponentiatedGradient.

    Notes
    -----
    - Aligns with Training Module requirement #1.
    - API mirrors Fairlearn and sklearn. `fit` requires `sensitive_features`.
    """

    base_estimator: Any
    constraint: Any
    eps: float = 0.01
    T: int = 50
    kwargs: Optional[dict] = None

    def __post_init__(self):
        self._wrapper_: Optional[ExponentiatedGradient] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y, sensitive_features):
        base = clone(self.base_estimator)
        self._wrapper_ = ExponentiatedGradient(
            estimator=base,
            constraints=self.constraint,
            eps=self.eps,
            **(self.kwargs or {}),
        )
        self._wrapper_.fit(X, y, sensitive_features=sensitive_features)
        # classes_ for sklearn compatibility
        try:
            self.classes_ = np.array(self._wrapper_.estimators_[0].classes_)  # type: ignore[attr-defined]
        except Exception:
            # Fallback for models that set classes_ on the wrapper
            self.classes_ = getattr(self._wrapper_, "classes_", None)
        return self

    def predict(self, X):
        if self._wrapper_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self._wrapper_.predict(X)

    def predict_proba(self, X):
        if self._wrapper_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        # 1) If the wrapper itself exposes predict_proba (unlikely), use it.
        if hasattr(self._wrapper_, "predict_proba"):
            return self._wrapper_.predict_proba(X)  # type: ignore[attr-defined]

        # 2) Fairlearn ExponentiatedGradient exposes a mixture:
        #    predictors_: list[estimator], weights_: np.ndarray of same length
        if hasattr(self._wrapper_, "predictors_") and hasattr(self._wrapper_, "weights_"):
            predictors = getattr(self._wrapper_, "predictors_")
            weights = np.asarray(getattr(self._wrapper_, "weights_"), dtype=float)

            if len(predictors) == 0:
                raise AttributeError("No predictors_ found on the reductions wrapper.")

            # Normalize just in case (defensive)
            if weights.ndim != 1 or len(weights) != len(predictors):
                raise AttributeError("weights_ shape mismatch on the reductions wrapper.")
            wsum = weights.sum()
            if wsum <= 0:
                # fallback to uniform average
                weights = np.ones_like(weights) / len(weights)
            else:
                weights = weights / wsum

            # Weighted average of per-estimator probabilities (binary or multiclass)
            probs = None
            for w, est in zip(weights, predictors):
                if not hasattr(est, "predict_proba"):
                    # Fallback: try decision_function -> map to prob via logistic if binary
                    if hasattr(est, "decision_function"):
                        df = est.decision_function(X)
                        # If binary, map logits to prob for positive class
                        if df.ndim == 1:
                            est_probs = np.column_stack(
                                [1.0 / (1.0 + np.exp(df)), 1.0 / (1.0 + np.exp(-df))]
                            )
                        else:
                            # multiclass decision_function: softmax
                            exps = np.exp(df - df.max(axis=1, keepdims=True))
                            est_probs = exps / exps.sum(axis=1, keepdims=True)
                    else:
                        # last resort: use hard predictions and one-hot
                        preds = est.predict(X)
                        n_classes = np.unique(preds).size
                        est_probs = np.zeros((len(preds), n_classes), dtype=float)
                        est_probs[np.arange(len(preds)), preds.astype(int)] = 1.0
                else:
                    est_probs = est.predict_proba(X)

                probs = (w * est_probs) if probs is None else (probs + w * est_probs)

            return probs

        # 3) If none of the above works, raise a clear error
        raise AttributeError(
            "predict_proba not available; underlying Fairlearn wrapper "
            "does not expose predictors_/weights_."
        )
