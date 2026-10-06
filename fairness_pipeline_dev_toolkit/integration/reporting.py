"""
Human-readable reporting utilities.

We expose a single `to_markdown_report(results)` that accepts the same mapping
you would pass to the MLflow logger. Useful for CLI, PR comments, and CI logs.

Also includes comprehensive training fairness report generation.
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


def format_ci_note_plain(ci_note: Optional[str]) -> Optional[str]:
    """Turn a machine-readable ``ci_note`` into plain words for reports/CLI.

    ``undefined:no_calibrated_interval (... #63)`` becomes
    ``no calibrated interval for this metric yet, see #63``.
    """
    if not ci_note:
        return None
    issue_m = re.search(r"(?:#|issues/)(\d+)", ci_note)
    issue = f"#{issue_m.group(1)}" if issue_m else None
    if ci_note.startswith("undefined:no_calibrated_interval"):
        base = "no calibrated interval for this metric yet"
        return f"{base}, see {issue}" if issue else base
    if ci_note.startswith("undefined:"):
        rest = ci_note[len("undefined:") :]
        # Keep parenthetical detail; soften underscores in the reason token.
        if " (" in rest:
            reason, detail = rest.split(" (", 1)
            plain = f"{reason.replace('_', ' ')} ({detail}"
        else:
            plain = rest.replace("_", " ")
        return f"{plain} (see {issue})" if issue and issue not in plain else plain
    return ci_note


def interpret_gap_interval(
    ci: Optional[Sequence[float]],
    *,
    ci_kind: Optional[str] = None,
    level: float = 0.95,
    delta: Optional[float] = None,
    p_value: Optional[float] = None,
) -> str:
    """Plain-language reading of a gap confidence interval.

    For ``ci_kind="simultaneous_pairwise"``:

    - ``L > 0``: largest gap is at least L at the stated level.
    - ``L = 0``: consistent with no gap; could be as large as U.
    - ``U < δ``: below δ with (level) confidence (equivalence wording).

    For ``"percentile"`` or ``"bca"``, the interval is labeled by kind and noted
    as not calibrated near zero. ``significant`` only when ``p_value`` is
    provided and ≤ 0.05.
    """
    if ci is None or len(ci) < 2 or ci[0] is None or ci[1] is None:
        return ""
    lo, hi = float(ci[0]), float(ci[1])
    pct = f"{level * 100:.0f}%"
    parts: List[str] = []
    kind = (ci_kind or "").strip() or None
    if kind == "simultaneous_pairwise":
        if lo > 0:
            parts.append(
                f"the largest gap is at least {lo:.4f} "
                f"({pct} simultaneous interval [{lo:.4f}, {hi:.4f}])"
            )
        else:
            parts.append(
                f"consistent with no gap; could be as large as {hi:.4f} "
                f"({pct} simultaneous interval [{lo:.4f}, {hi:.4f}])"
            )
        if delta is not None and hi < delta:
            parts.append(f"below {delta:g} with {pct} confidence")
    elif kind in ("percentile", "bca"):
        parts.append(f"{pct} {kind} interval [{lo:.4f}, {hi:.4f}] (not calibrated near zero)")
        if delta is not None and hi < delta:
            parts.append(f"upper bound < {delta:g}")
    else:
        label = kind or "confidence"
        parts.append(f"{pct} {label} interval [{lo:.4f}, {hi:.4f}]")
        if delta is not None and hi < delta:
            parts.append(f"upper bound < {delta:g}")
    if p_value is not None:
        if p_value <= 0.05:
            parts.append(f"permutation p-value {p_value:.4g} (significant at 0.05)")
        else:
            parts.append(f"permutation p-value {p_value:.4g} (not significant at 0.05)")
    return "; ".join(parts)


def _fmt_ci(ci, ci_note: Optional[str] = None, *, level: float = 0.95) -> str:
    """Format a CI cell: numeric bounds, or the plain-words ``ci_note`` reason."""
    _ = level  # reserved for callers that label the column with the level
    if ci is not None and len(ci) >= 2 and ci[0] is not None and ci[1] is not None:
        try:
            return f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"
        except (TypeError, ValueError):
            pass
    plain = format_ci_note_plain(ci_note)
    if plain:
        return plain
    return "—"


def _coerce(val: Any) -> Dict[str, Any]:
    if is_dataclass(val):
        return asdict(val)
    if isinstance(val, dict):
        return val
    return {"value": val}


def to_markdown_report(
    results: Mapping[str, Any],
    *,
    title: str = "Fairness Report",
    ci_level: float = 0.95,
) -> str:
    """
    Convert a metrics mapping into a Markdown document with a summary table.

    Undefined CIs render the plain-words ``ci_note`` (never the literal ``None``).

    Returns
    -------
    str
        Markdown content suitable for PR comments, artifacts, or CLI output.
    """
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"# {title}", "", f"_Generated: {ts}_", ""]

    level_label = f"{ci_level * 100:.0f}%"
    # Summary table
    lines.append(f"| Metric | Value | CI ({level_label}) | Effect Size | p_value | n_per_group |")
    lines.append("|---|---:|---|---:|---:|---|")

    for name, val in results.items():
        item = _coerce(val)
        value = item.get("value", "—")
        if isinstance(value, float):
            value = f"{value:.6f}"
        ci = _fmt_ci(item.get("ci"), item.get("ci_note"), level=ci_level)
        eff = item.get("effect_size")
        if isinstance(eff, float):
            eff = f"{eff:.6f}"
        elif eff is None:
            eff = "—"
        p_val = item.get("p_value")
        if isinstance(p_val, float):
            p_display = f"{p_val:.4g}"
        else:
            p_display = "—"
        n_per_group = item.get("n_per_group")
        n_display = json.dumps(n_per_group) if n_per_group else "—"
        caveat = item.get("caveat")
        if caveat:
            value = f"{value}*"
        lines.append(f"| `{name}` | {value} | {ci} | {eff} | {p_display} | {n_display} |")

    lines.append("")
    lines.append(
        "> Note: when a CI is undefined, the CI column shows the reason "
        "(e.g. “no calibrated interval for this metric yet, see #63”), never `None`. "
        "`—` means the field was not computed. Use `p_value` for significance claims."
    )

    caveats = []
    for name, val in results.items():
        item = _coerce(val)
        text = item.get("caveat")
        if text:
            caveats.append(f"- `{name}`: {text}")
    if caveats:
        lines.append("")
        lines.append("* Demo / non-evidential result — see caveats below.")
        lines.append("")
        lines.append("## Caveats")
        lines.append("")
        lines.extend(caveats)

    return "\n".join(lines)


# ============================================================================
# Training Fairness Report Generation
# ============================================================================


def _compute_group_rates(y_pred, sensitive, min_group_size: int = 30) -> Dict[str, float]:
    """Extract per-group positive prediction rates."""
    import numpy as np
    import pandas as pd

    yp = np.asarray(y_pred)
    sens = np.asarray(sensitive)
    s = pd.Series(sens)
    counts = s.value_counts()
    valid = s.map(counts) >= min_group_size
    s_valid = s[valid].to_numpy()
    yp_valid = yp[valid]

    rates = {}
    groups = np.unique(s_valid)
    for g in groups:
        m = s_valid == g
        if m.sum() >= min_group_size:
            rates[str(g)] = float(yp_valid[m].mean())
    return rates


def _assess_severity(value: float, threshold: float) -> Tuple[str, str]:
    """Categorize fairness issue severity."""
    if value > 0.10:
        return "Critical", "Exceeds threshold by >100%. Immediate action required."
    elif value > threshold:
        excess_pct = ((value - threshold) / threshold) * 100
        return (
            "High",
            f"Exceeds threshold by {excess_pct:.1f}%. Requires mitigation before deployment.",
        )
    elif value > 0.02:
        return "Medium", "Within acceptable range but should be monitored."
    else:
        return "Low", "Meets fairness standards."


def _interpret_metric_value(
    value: float, metric_name: str, threshold: Optional[float] = None
) -> str:
    """Provide plain-language interpretation of metric values."""
    if metric_name == "demographic_parity_difference":
        interpretation = (
            f"Demographic Parity Difference measures the gap in positive prediction rates between groups. "
            f"A value of {value:.4f} means the highest-rate group receives {value*100:.2f} percentage points "
            f"more positive predictions than the lowest-rate group."
        )
        if threshold is not None:
            if value <= 0.02:
                interpretation += " This is considered excellent (≤2%)."
            elif value <= threshold:
                interpretation += f" This is acceptable (≤{threshold*100:.0f}%)."
            else:
                interpretation += (
                    f" This exceeds the threshold of {threshold*100:.0f}% and is concerning."
                )
        return interpretation
    elif metric_name == "equalized_odds_difference":
        interpretation = (
            f"Equalized Odds Difference measures the maximum gap in either true positive rates or "
            f"false positive rates across groups. A value of {value:.4f} indicates a {value*100:.2f} "
            f"percentage point difference in error rates between groups."
        )
        return interpretation
    return f"Metric value: {value:.4f}"


def _analyze_training_convergence(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze Lagrangian training convergence."""
    if not history or len(history) < 2:
        return {
            "converged": False,
            "violation_trend": "insufficient_data",
            "lambda_trend": "insufficient_data",
        }

    violations = [h.get("violation", 0) for h in history]
    lambdas = [h.get("lambda", 0) for h in history]

    # Check if violations decreased
    if violations[-1] < violations[0] * 0.5:
        violation_trend = "improving"
    elif violations[-1] < violations[0]:
        violation_trend = "slight_improvement"
    elif violations[-1] > violations[0] * 1.5:
        violation_trend = "worsening"
    else:
        violation_trend = "stable"

    # Check if lambda increased (should increase when constraints violated)
    if lambdas[-1] > lambdas[0] * 1.5:
        lambda_trend = "increasing"
    elif lambdas[-1] > lambdas[0]:
        lambda_trend = "slight_increase"
    else:
        lambda_trend = "stable"

    converged = violation_trend in ["improving", "slight_improvement"] and lambda_trend in [
        "increasing",
        "slight_increase",
    ]

    return {
        "converged": converged,
        "violation_trend": violation_trend,
        "lambda_trend": lambda_trend,
        "initial_violation": violations[0],
        "final_violation": violations[-1],
        "initial_lambda": lambdas[0],
        "final_lambda": lambdas[-1],
    }


def _generate_recommendations(report_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate lifecycle-stage-specific actionable recommendations."""
    recommendations = {
        "data_stage": [],
        "training_stage": [],
        "evaluation_stage": [],
        "deployment_stage": [],
    }

    # Data stage recommendations
    data_stage = report_data.get("data_stage", {})
    rep_bias = data_stage.get("representation_bias", {})
    for attr, result in rep_bias.items():
        if hasattr(result, "proportions") and result.proportions:
            props = result.proportions
            min_prop = min(props.values())
            max_prop = max(props.values())
            if max_prop / min_prop > 2:
                min_group = min(props.items(), key=lambda x: x[1])
                recommendations["data_stage"].append(
                    f"Collect more balanced data for underrepresented groups. "
                    f"'{min_group[0]}' represents only {min_group[1]*100:.1f}% of the sample."
                )

    proxies = data_stage.get("proxy_variables", {})
    for attr, proxy_list in proxies.items():
        for proxy in proxy_list:
            if hasattr(proxy, "flagged") and proxy.flagged:
                recommendations["data_stage"].append(
                    f"Remove or transform feature '{proxy.feature}' which proxies for {attr} "
                    f"(strength: {proxy.strength:.3f}, measure: {proxy.measure})."
                )

    disparities = data_stage.get("statistical_disparities", {})
    for attr, disp_list in disparities.items():
        for disp in disp_list:
            if hasattr(disp, "flagged") and disp.flagged:
                recommendations["data_stage"].append(
                    f"Investigate feature '{disp.feature}' showing significant differences across {attr} groups "
                    f"(p={disp.pvalue:.4f}, test={disp.test})."
                )

    # Training stage recommendations
    final_metrics = report_data.get("final_metrics", {})
    threshold = report_data.get("metadata", {}).get("fairness_threshold", 0.05)
    dp_result = final_metrics.get("demographic_parity")
    if dp_result and hasattr(dp_result, "value") and dp_result.value > threshold:
        recommendations["training_stage"].append(
            "Increase fairness constraint strength: reduce `dp_tolerance` from 0.02 to 0.01 "
            "or increase `lambda_lr` from 0.01 to 0.02 to better enforce demographic parity."
        )

    mitigation = report_data.get("mitigation", {})
    lagrangian = mitigation.get("lagrangian_training", {})
    convergence = lagrangian.get("convergence", {})
    if not convergence.get("converged", False):
        if convergence.get("violation_trend") == "worsening":
            recommendations["training_stage"].append(
                "Fairness violations increased during training. Model may need more epochs, "
                "stronger fairness penalty, or different constraint formulation."
            )
        elif convergence.get("lambda_trend") == "stable":
            recommendations["training_stage"].append(
                "Lagrangian multiplier not increasing. Consider increasing `lambda_lr` or "
                "tightening `dp_tolerance` to enforce constraints more strongly."
            )

    # Evaluation stage recommendations
    if dp_result is not None:
        ci = getattr(dp_result, "ci", None)
        ci_note = getattr(dp_result, "ci_note", None)
        p_value = getattr(dp_result, "p_value", None)
        ci_kind = getattr(dp_result, "ci_kind", None)
        if ci is None and ci_note:
            plain = format_ci_note_plain(ci_note)
            if plain:
                recommendations["evaluation_stage"].append(
                    f"Confidence interval undefined: {plain}."
                )
        elif ci is not None and len(ci) >= 2 and ci[0] is not None and ci[1] is not None:
            reading = interpret_gap_interval(
                ci,
                ci_kind=ci_kind,
                level=0.95,
                delta=threshold,
                p_value=p_value,
            )
            if reading:
                recommendations["evaluation_stage"].append(reading + ".")

    if dp_result and hasattr(dp_result, "effect_size") and dp_result.effect_size:
        if dp_result.effect_size > 1.5:
            recommendations["evaluation_stage"].append(
                f"Risk ratio of {dp_result.effect_size:.2f} indicates substantial disparity "
                "requiring immediate attention."
            )

    # Deployment stage recommendations
    threshold_status = report_data.get("comparison", {}).get("threshold_status", "unknown")
    if threshold_status == "fail":
        dp_val = dp_result.value if dp_result and hasattr(dp_result, "value") else "N/A"
        recommendations["deployment_stage"].append(
            f"⚠️ DO NOT DEPLOY without additional mitigation. Current DP difference ({dp_val:.4f}) "
            f"exceeds policy threshold ({threshold:.4f})."
        )
    elif threshold_status == "pass":
        if dp_result and hasattr(dp_result, "value"):
            if dp_result.value > threshold * 0.8:  # Close to threshold
                recommendations["deployment_stage"].append(
                    "Deploy with monitoring. Set up drift detection for fairness metrics to "
                    "ensure continued compliance."
                )
            else:
                recommendations["deployment_stage"].append(
                    "Safe to deploy. Establish baseline monitoring metrics for ongoing fairness tracking."
                )

    return recommendations


def _prepare_report_data(
    raw_data: Dict[str, Any],
    *,
    compute_performance: bool = True,
    compute_convergence: bool = True,
) -> Dict[str, Any]:
    """
    Prepare complete report_data dict from raw inputs.

    Automatically computes performance metrics and convergence analysis if raw data provided.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Start with a copy of the input data
    report_data = raw_data.copy()

    # Auto-compute performance metrics if y_true/y_pred provided
    if compute_performance:
        y_true = raw_data.get("y_true")
        y_pred = raw_data.get("y_pred")
        if y_true is not None and y_pred is not None and "model_performance" not in report_data:
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            val_acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            report_data["model_performance"] = {
                "accuracy": float(val_acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    # Auto-compute convergence analysis if training_history provided
    if compute_convergence:
        mitigation = report_data.get("mitigation", {})
        lagrangian = mitigation.get("lagrangian_training", {})
        training_history = lagrangian.get("history") or raw_data.get("training_history")

        if training_history and "convergence" not in lagrangian:
            convergence_analysis = _analyze_training_convergence(training_history)
            if "lagrangian_training" not in mitigation:
                mitigation["lagrangian_training"] = {}
            mitigation["lagrangian_training"]["convergence"] = convergence_analysis
            mitigation["lagrangian_training"]["final_lambda"] = (
                training_history[-1].get("lambda") if training_history else None
            )
            report_data["mitigation"] = mitigation

    # Auto-compute weight statistics if sample_weights provided
    mitigation = report_data.get("mitigation", {})
    reweighting = mitigation.get("instance_reweighting", {})
    sample_weights = reweighting.get("sample_weights")
    if sample_weights is None:
        sample_weights = raw_data.get("sample_weights")

    if sample_weights is not None and "weights_stats" not in reweighting:
        sample_weights = np.asarray(sample_weights)
        weights_min = float(sample_weights.min())
        weights_max = float(sample_weights.max())
        weights_mean = float(sample_weights.mean())
        weight_ratio = weights_max / weights_min if weights_min > 0 else float("inf")

        if "instance_reweighting" not in mitigation:
            mitigation["instance_reweighting"] = {}
        mitigation["instance_reweighting"]["weights_stats"] = {
            "min": weights_min,
            "max": weights_max,
            "mean": weights_mean,
        }
        mitigation["instance_reweighting"]["impact"] = (
            "high" if weight_ratio > 10 else "moderate" if weight_ratio > 5 else "low"
        )
        report_data["mitigation"] = mitigation

    # Ensure metadata has timestamp if not provided
    if "metadata" not in report_data:
        report_data["metadata"] = {}
    if "timestamp" not in report_data["metadata"]:
        report_data["metadata"]["timestamp"] = datetime.datetime.utcnow()

    return report_data


def generate_training_fairness_report(
    report_data: Dict[str, Any],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    compute_performance: bool = True,
    compute_convergence: bool = True,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Path]]]:
    """
    Generate comprehensive training fairness report in Markdown and JSON formats.

    Parameters
    ----------
    report_data : Dict[str, Any]
        Structured dictionary containing all training and fairness data.
        Expected keys:
        - metadata: timestamp (auto-set if missing), model_name, sensitive_attributes, fairness_threshold
        - data_stage: representation_bias, statistical_disparities, proxy_variables
        - baseline_metrics: Dict of baseline metric results
        - mitigation: instance_reweighting (with sample_weights or weights_stats), lagrangian_training (with history or convergence)
        - final_metrics: Dict of final metric results
        - model_performance: accuracy, precision, recall, f1 (auto-computed if y_true/y_pred provided)
        - comparison: improvement, threshold_status
        - y_true, y_pred: Optional raw labels/predictions for auto-computing performance metrics
        - training_history: Optional raw training history for auto-computing convergence
        - sample_weights: Optional raw sample weights for auto-computing weight statistics
    output_dir : Optional[Union[str, Path]], optional
        Directory to save report files. If provided, saves markdown and JSON files.
    compute_performance : bool, default=True
        If True and y_true/y_pred provided, automatically compute performance metrics.
    compute_convergence : bool, default=True
        If True and training_history provided, automatically compute convergence analysis.

    Returns
    -------
    Tuple[str, Dict[str, Any], Optional[Dict[str, Path]]]
        Markdown report string, JSON-serializable dictionary, and optional file paths dict.
        File paths dict is None if output_dir not provided, otherwise contains 'markdown' and 'json' keys.
    """
    from dataclasses import asdict

    import numpy as np

    # Prepare report data (auto-compute metrics, convergence, etc.)
    report_data = _prepare_report_data(
        report_data,
        compute_performance=compute_performance,
        compute_convergence=compute_convergence,
    )

    metadata = report_data.get("metadata", {})
    timestamp = metadata.get("timestamp", datetime.datetime.utcnow())
    if isinstance(timestamp, str):
        ts_str = timestamp
    else:
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = []
    lines.append("# Training Fairness Report")
    lines.append("")
    lines.append(f"_Generated: {ts_str}_")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    final_metrics = report_data.get("final_metrics", {})
    threshold = metadata.get("fairness_threshold", 0.05)
    dp_result = final_metrics.get("demographic_parity")
    eo_result = final_metrics.get("equalized_odds")

    threshold_status = report_data.get("comparison", {}).get("threshold_status", "unknown")
    if threshold_status == "pass":
        status_badge = "✅ **PASS**"
    elif threshold_status == "fail":
        status_badge = "❌ **FAIL**"
    else:
        status_badge = "⚠️ **WARN**"

    lines.append(f"**Overall Status:** {status_badge}")
    lines.append("")

    # Key metrics at a glance
    lines.append("**Key Metrics:**")
    if dp_result and hasattr(dp_result, "value"):
        dp_val = dp_result.value
        lines.append(
            f"- Demographic Parity Difference: **{dp_val:.4f}** (threshold: {threshold:.4f})"
        )
    if eo_result and hasattr(eo_result, "value"):
        eo_val = eo_result.value
        lines.append(f"- Equalized Odds Difference: **{eo_val:.4f}**")
    lines.append("")

    # Most critical issue
    if dp_result and hasattr(dp_result, "value") and dp_result.value > threshold:
        excess = ((dp_result.value - threshold) / threshold) * 100
        lines.append(
            f"**Critical Issue:** Race-based demographic parity exceeds threshold by {excess:.1f}% "
            f"({dp_result.value:.4f} vs {threshold:.4f})."
        )
    lines.append("")

    # Top recommendation
    recommendations = _generate_recommendations(report_data)
    if recommendations.get("deployment_stage"):
        lines.append(f"**Top Recommendation:** {recommendations['deployment_stage'][0]}")
    elif recommendations.get("training_stage"):
        lines.append(f"**Top Recommendation:** {recommendations['training_stage'][0]}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 1: Data Quality & Bias Detection
    lines.append("## 1. Data Quality & Bias Detection")
    lines.append("")

    data_stage = report_data.get("data_stage", {})
    rep_bias = data_stage.get("representation_bias", {})
    if rep_bias:
        lines.append("### Representation Analysis")
        lines.append("")
        for attr, result in rep_bias.items():
            lines.append(f"**{attr.capitalize()}:**")
            if hasattr(result, "proportions") and result.proportions:
                props = result.proportions
                for group, prop in sorted(props.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"- {group}: {prop*100:.2f}%")
                min_prop = min(props.values())
                max_prop = max(props.values())
                ratio = max_prop / min_prop if min_prop > 0 else float("inf")
                if ratio > 2:
                    severity = "severe"
                elif ratio > 1.5:
                    severity = "moderate"
                else:
                    severity = "minor"
                lines.append(f"  _Imbalance severity: {severity} (max/min ratio: {ratio:.2f})_")
            lines.append("")

    disparities = data_stage.get("statistical_disparities", {})
    if disparities:
        lines.append("### Statistical Disparities")
        lines.append("")
        for attr, disp_list in disparities.items():
            flagged = [d for d in disp_list if hasattr(d, "flagged") and d.flagged]
            if flagged:
                lines.append(f"**{attr.capitalize()}:** {len(flagged)} features flagged")
                for disp in flagged[:5]:  # Show top 5
                    lines.append(
                        f"- `{disp.feature}`: {disp.test} test, p={disp.pvalue:.4f} "
                        f"({'significant' if disp.flagged else 'not significant'})"
                    )
            else:
                lines.append(f"**{attr.capitalize()}:** No significant disparities detected")
            lines.append("")

    proxies = data_stage.get("proxy_variables", {})
    if proxies:
        lines.append("### Proxy Variables")
        lines.append("")
        for attr, proxy_list in proxies.items():
            flagged = [p for p in proxy_list if hasattr(p, "flagged") and p.flagged]
            if flagged:
                lines.append(f"**{attr.capitalize()}:** {len(flagged)} proxies flagged")
                for proxy in flagged[:5]:
                    lines.append(
                        f"- `{proxy.feature}`: {proxy.measure}={proxy.strength:.3f} "
                        f"({'strong proxy' if proxy.flagged else 'weak correlation'})"
                    )
            else:
                lines.append(f"**{attr.capitalize()}:** No strong proxy variables detected")
            lines.append("")

    lines.append("---")
    lines.append("")

    # Section 2: Baseline Fairness Assessment
    lines.append("## 2. Baseline Fairness Assessment")
    lines.append("")

    baseline_metrics = report_data.get("baseline_metrics", {})
    for key, result in baseline_metrics.items():
        if hasattr(result, "value") and not np.isnan(result.value):
            metric_name = key.replace("_demographic_parity", "").replace("_", " ").title()
            lines.append(f"### {metric_name}")
            lines.append("")
            lines.append(
                _interpret_metric_value(result.value, "demographic_parity_difference", threshold)
            )
            lines.append("")

            if hasattr(result, "n_per_group") and result.n_per_group:
                # Compute group rates (we'd need y_pred and sensitive, but we'll approximate from n_per_group)
                lines.append("**Group-level breakdowns:**")
                lines.append("| Group | Sample Size |")
                lines.append("|---|---|")
                for group, n in result.n_per_group.items():
                    lines.append(f"| {group} | {n} |")
                lines.append("")

            if hasattr(result, "ci") and result.ci:
                reading = interpret_gap_interval(
                    result.ci,
                    ci_kind=getattr(result, "ci_kind", None),
                    level=0.95,
                    delta=threshold,
                    p_value=getattr(result, "p_value", None),
                )
                lines.append(f"**Confidence Interval:** {reading}.")
                lines.append("")
            elif getattr(result, "ci_note", None):
                plain = format_ci_note_plain(result.ci_note)
                lines.append(f"**Confidence Interval:** {plain}.")
                lines.append("")

    lines.append("---")
    lines.append("")

    # Section 3: Mitigation Strategy Applied
    lines.append("## 3. Mitigation Strategy Applied")
    lines.append("")

    mitigation = report_data.get("mitigation", {})
    reweighting = mitigation.get("instance_reweighting", {})
    if reweighting:
        lines.append("### Instance Reweighting")
        lines.append("")
        weights_stats = reweighting.get("weights_stats", {})
        if weights_stats:
            min_w = weights_stats.get("min", 0)
            max_w = weights_stats.get("max", 0)
            mean_w = weights_stats.get("mean", 1.0)
            ratio = max_w / min_w if min_w > 0 else float("inf")
            lines.append(f"- Weight range: {min_w:.3f} to {max_w:.3f} (mean: {mean_w:.3f})")
            if ratio > 10:
                lines.append(
                    f"  _Weight range of {ratio:.1f}x indicates significant rebalancing needed._"
                )
            elif ratio > 5:
                lines.append(f"  _Weight range of {ratio:.1f}x indicates moderate rebalancing._")
            else:
                lines.append(
                    f"  _Weight range of {ratio:.1f}x indicates minimal rebalancing needed._"
                )
        lines.append("")

    lagrangian = mitigation.get("lagrangian_training", {})
    if lagrangian:
        lines.append("### Lagrangian Training")
        lines.append("")
        convergence = lagrangian.get("convergence", {})
        if convergence:
            if convergence.get("converged"):
                lines.append("✅ **Training converged successfully**")
            else:
                lines.append("⚠️ **Training did not fully converge**")
            lines.append("")
            lines.append(f"- Violation trend: {convergence.get('violation_trend', 'unknown')}")
            lines.append(f"- Lambda trend: {convergence.get('lambda_trend', 'unknown')}")
            if convergence.get("initial_violation") is not None:
                lines.append(
                    f"- Violation: {convergence['initial_violation']:.4f} → "
                    f"{convergence['final_violation']:.4f}"
                )
            if convergence.get("initial_lambda") is not None:
                lines.append(
                    f"- Lambda: {convergence['initial_lambda']:.4f} → {convergence['final_lambda']:.4f}"
                )
        lines.append("")

    lines.append("---")
    lines.append("")

    # Section 4: Final Fairness Evaluation
    lines.append("## 4. Final Fairness Evaluation")
    lines.append("")

    if dp_result and hasattr(dp_result, "value"):
        lines.append("### Demographic Parity")
        lines.append("")
        lines.append(
            _interpret_metric_value(dp_result.value, "demographic_parity_difference", threshold)
        )
        lines.append("")

        severity, severity_msg = _assess_severity(dp_result.value, threshold)
        lines.append(f"**Severity:** {severity} - {severity_msg}")
        lines.append("")

        if hasattr(dp_result, "n_per_group") and dp_result.n_per_group:
            lines.append("**Group-level rates:**")
            lines.append("| Group | Sample Size |")
            lines.append("|---|---|")
            for group, n in dp_result.n_per_group.items():
                lines.append(f"| {group} | {n} |")
            lines.append("")

        if hasattr(dp_result, "ci") and dp_result.ci:
            reading = interpret_gap_interval(
                dp_result.ci,
                ci_kind=getattr(dp_result, "ci_kind", None),
                level=0.95,
                delta=threshold,
                p_value=getattr(dp_result, "p_value", None),
            )
            lines.append(f"**Confidence Interval:** {reading}.")
            lines.append("")
        elif getattr(dp_result, "ci_note", None):
            lines.append(f"**Confidence Interval:** {format_ci_note_plain(dp_result.ci_note)}.")
            lines.append("")

        if hasattr(dp_result, "effect_size") and dp_result.effect_size:
            lines.append(f"**Effect Size (Risk Ratio):** {dp_result.effect_size:.3f}")
            lines.append("")

    if eo_result and hasattr(eo_result, "value"):
        lines.append("### Equalized Odds")
        lines.append("")
        lines.append(_interpret_metric_value(eo_result.value, "equalized_odds_difference"))
        lines.append("")

        if hasattr(eo_result, "ci") and eo_result.ci:
            reading = interpret_gap_interval(
                eo_result.ci,
                ci_kind=getattr(eo_result, "ci_kind", None),
                level=0.95,
                delta=threshold,
                p_value=getattr(eo_result, "p_value", None),
            )
            lines.append(f"**Confidence Interval:** {reading}.")
            lines.append("")
        elif getattr(eo_result, "ci_note", None):
            lines.append(f"**Confidence Interval:** {format_ci_note_plain(eo_result.ci_note)}.")
            lines.append("")

    # Comparison to baseline
    comparison = report_data.get("comparison", {})
    improvement = comparison.get("improvement")
    if improvement is not None:
        lines.append("### Comparison to Baseline")
        lines.append("")
        if improvement < 0:
            lines.append(
                f"✅ **Improvement:** Fairness improved by {abs(improvement):.4f} (reduction in unfairness)"
            )
        elif improvement > 0:
            lines.append(
                f"⚠️ **Regression:** Fairness worsened by {improvement:.4f} (increase in unfairness)"
            )
        else:
            lines.append("➡️ **No change:** Fairness metrics unchanged")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Section 5: Actionable Recommendations
    lines.append("## 5. Actionable Recommendations")
    lines.append("")

    for stage in ["data_stage", "training_stage", "evaluation_stage", "deployment_stage"]:
        stage_name = stage.replace("_", " ").title()
        recs = recommendations.get(stage, [])
        if recs:
            lines.append(f"### {stage_name}")
            lines.append("")
            for i, rec in enumerate(recs, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        else:
            lines.append(f"### {stage_name}")
            lines.append("")
            lines.append("No specific recommendations at this stage.")
            lines.append("")

    lines.append("---")
    lines.append("")

    # Section 6: Model Performance Context
    lines.append("## 6. Model Performance Context")
    lines.append("")

    perf = report_data.get("model_performance", {})
    if perf:
        lines.append("**Performance Metrics:**")
        if "accuracy" in perf:
            lines.append(f"- Accuracy: {perf['accuracy']:.4f}")
        if "precision" in perf:
            lines.append(f"- Precision: {perf['precision']:.4f}")
        if "recall" in perf:
            lines.append(f"- Recall: {perf['recall']:.4f}")
        if "f1" in perf:
            lines.append(f"- F1 Score: {perf['f1']:.4f}")
        lines.append("")

    # Helper function to safely convert to dict
    def _safe_to_dict(obj):
        if is_dataclass(obj):
            return asdict(obj)
        elif hasattr(obj, "__dict__"):
            return {k: _safe_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: _safe_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_safe_to_dict(item) for item in obj]
        else:
            return obj

    # Generate JSON structure
    json_data = {
        "metadata": metadata,
        "executive_summary": {
            "status": threshold_status,
            "key_metrics": {
                "demographic_parity_difference": (
                    dp_result.value if dp_result and hasattr(dp_result, "value") else None
                ),
                "equalized_odds_difference": (
                    eo_result.value if eo_result and hasattr(eo_result, "value") else None
                ),
            },
            "threshold": threshold,
        },
        "data_stage": {
            "representation_bias": {k: _safe_to_dict(v) for k, v in rep_bias.items()},
            "statistical_disparities": {
                k: [_safe_to_dict(d) for d in v] for k, v in disparities.items()
            },
            "proxy_variables": {k: [_safe_to_dict(p) for p in v] for k, v in proxies.items()},
        },
        "baseline_metrics": {k: _safe_to_dict(v) for k, v in baseline_metrics.items()},
        "mitigation": _safe_to_dict(mitigation),
        "final_metrics": {k: _safe_to_dict(v) for k, v in final_metrics.items()},
        "model_performance": perf,
        "comparison": comparison,
        "recommendations": recommendations,
    }

    markdown_report = "\n".join(lines)

    # Handle file I/O if output_dir provided
    file_paths = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_md_path = output_path / "training_fairness_report.md"
        report_json_path = output_path / "training_fairness_report.json"

        # Save markdown
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)

        # Save JSON
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, default=str, ensure_ascii=False)

        file_paths = {
            "markdown": report_md_path,
            "json": report_json_path,
        }

    return markdown_report, json_data, file_paths
