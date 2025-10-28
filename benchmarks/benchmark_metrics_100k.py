import time
import numpy as np
import pandas as pd
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer

def make_fake_classification(n=100_000, seed=7):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    # 5-way sensitive attribute with imbalanced groups
    sensitive = rng.choice(['A', 'B', 'C', 'D', 'E'], size=n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    # attrs_df for intersectional testing
    attrs = pd.DataFrame({
        "race": sensitive,
        "gender": rng.choice(['M', 'F'], size=n, p=[0.6, 0.4]),
        "age_group": rng.choice(['<30', '30-50', '50+'], size=n, p=[0.3, 0.5, 0.2]),
    })
    y_reg = rng.normal(0, 1, size=n)  # for regression testing
    y_hat = y_true + rng.normal(0, 0.5, size=n) # mock continuous predictions
    return y_true, y_pred, sensitive, attrs, y_reg, y_hat

if __name__ == "__main__":
    y_true, y_pred, sensitive, attrs, y_reg, y_hat = make_fake_classification()

    fa = FairnessAnalyzer(min_group_size=50, backend="native")

    start = time.time()
    dp_result = fa.demographic_parity_difference(y_pred, sensitive)
    end = time.time()
    print(f"Demographic Parity Difference: {dp_result.value:.4f}, Time taken: {end - start:.2f} seconds")

    start = time.time()
    eo_result = fa.equalized_odds_difference(y_true, y_pred, sensitive)
    end = time.time()
    print(f"Equalized Odds Difference: {eo_result.value:.4f}, Time taken: {end - start:.2f} seconds")

    start = time.time()
    mae_parity_result = fa.mae_parity_difference(y_reg, y_hat, sensitive)
    end = time.time()
    print(f"MAE Parity Difference: {mae_parity_result.value:.4f}, Time taken: {end - start:.2f} seconds")

    start = time.time()
    dp_inter_result = fa.demographic_parity_difference(
        y_pred,
        sensitive=None,
        intersectional=True,
        attrs_df=attrs
    )
    end = time.time()
    print(f"Demographic Parity Difference (Intersectional): {dp_inter_result.value:.4f}, Time taken: {end - start:.2f} seconds")

    start = time.time()
    eo_inter_result = fa.equalized_odds_difference(
        y_true,
        y_pred,
        sensitive=None,
        intersectional=True,
        attrs_df=attrs
    )
    end = time.time()
    print(f"Equalized Odds Difference (Intersectional): {eo_inter_result.value:.4f}, Time taken: {end - start:.2f} seconds")

    start = time.time()
    mae_parity_result = fa.mae_parity_difference(
        y_reg,
        y_hat,
        sensitive=None,
        intersectional=True,
        attrs_df=attrs
    )
    end = time.time()
    print(f"MAE Parity Difference (Intersectional): {mae_parity_result.value:.4f}, Time taken: {end - start:.2f} seconds")