# Training Fairness Report

_Generated: 2026-01-14 14:07:21 UTC_

## Executive Summary

**Overall Status:** ❌ **FAIL**

**Key Metrics:**
- Demographic Parity Difference: **0.0659** (threshold: 0.0500)
- Equalized Odds Difference: **0.0608**

**Critical Issue:** Race-based demographic parity exceeds threshold by 31.8% (0.0659 vs 0.0500).

**Top Recommendation:** ⚠️ DO NOT DEPLOY without additional mitigation. Current DP difference (0.0659) exceeds policy threshold (0.0500).

---

## 1. Data Quality & Bias Detection

### Representation Analysis

**Race:**
- White: 35.02%
- Black: 25.44%
- Asian: 19.09%
- Hispanic: 15.41%
- Other: 5.04%
  _Imbalance severity: severe (max/min ratio: 6.95)_

**Gender:**
- Male: 48.64%
- Female: 47.19%
- Non-binary: 4.17%
  _Imbalance severity: severe (max/min ratio: 11.65)_

### Statistical Disparities

**Race:** 1 features flagged
- `hired`: chi2 test, p=0.0000 (significant)

**Gender:** No significant disparities detected

### Proxy Variables

**Race:** No strong proxy variables detected

**Gender:** No strong proxy variables detected

---

## 2. Baseline Fairness Assessment

### Race

Demographic Parity Difference measures the gap in positive prediction rates between groups. A value of 0.0424 means the highest-rate group receives 4.24 percentage points more positive predictions than the lowest-rate group. This is acceptable (≤5%).

**Group-level breakdowns:**
| Group | Sample Size |
|---|---|
| 0 | 5198 |
| 1 | 2802 |

**Confidence Interval:** 95% confident the true difference is between 0.0198 and 0.0629.

### Gender

Demographic Parity Difference measures the gap in positive prediction rates between groups. A value of 0.0170 means the highest-rate group receives 1.70 percentage points more positive predictions than the lowest-rate group. This is considered excellent (≤2%).

**Group-level breakdowns:**
| Group | Sample Size |
|---|---|
| 0 | 4109 |
| 1 | 3891 |

**Confidence Interval:** 95% confident the true difference is between 0.0014 and 0.0362.

---

## 3. Mitigation Strategy Applied

### Instance Reweighting

- Weight range: 0.391 to 31.636 (mean: 1.000)
  _Weight range of 81.0x indicates significant rebalancing needed._

### Lagrangian Training

⚠️ **Training did not fully converge**

- Violation trend: stable
- Lambda trend: increasing
- Violation: 0.0000 → 0.0000
- Lambda: 0.0002 → 0.0944

---

## 4. Final Fairness Evaluation

### Demographic Parity

Demographic Parity Difference measures the gap in positive prediction rates between groups. A value of 0.0659 means the highest-rate group receives 6.59 percentage points more positive predictions than the lowest-rate group. This exceeds the threshold of 5% and is concerning.

**Severity:** High - Exceeds threshold by 31.8%. Requires mitigation before deployment.

**Group-level rates:**
| Group | Sample Size |
|---|---|
| 0 | 1282 |
| 1 | 718 |

**95% Confidence Interval:** [0.0323, 0.0985]

**Effect Size (Risk Ratio):** nan

### Equalized Odds

Equalized Odds Difference measures the maximum gap in either true positive rates or false positive rates across groups. A value of 0.0608 indicates a 6.08 percentage point difference in error rates between groups.

**95% Confidence Interval:** [0.0371, 0.1244]

### Comparison to Baseline

✅ **Improvement:** Fairness improved by 0.0235 (reduction in unfairness)

---

## 5. Actionable Recommendations

### Data Stage

1. Collect more balanced data for underrepresented groups. 'Other' represents only 5.0% of the sample.
2. Collect more balanced data for underrepresented groups. 'Non-binary' represents only 4.2% of the sample.
3. Investigate feature 'hired' showing significant differences across race groups (p=0.0000, test=chi2).

### Training Stage

1. Increase fairness constraint strength: reduce `dp_tolerance` from 0.02 to 0.01 or increase `lambda_lr` from 0.01 to 0.02 to better enforce demographic parity.

### Evaluation Stage

No specific recommendations at this stage.

### Deployment Stage

1. ⚠️ DO NOT DEPLOY without additional mitigation. Current DP difference (0.0659) exceeds policy threshold (0.0500).

---

## 6. Model Performance Context

**Performance Metrics:**
- Accuracy: 0.7050
- Precision: 0.5734
- Recall: 0.2595
- F1 Score: 0.3573
