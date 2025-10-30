import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 200

# Synthetic binary classification data
df = pd.DataFrame(
    {
        "y_true": rng.integers(0, 2, n),  # ground truth labels
        "y_pred": rng.integers(0, 2, n),  # model predictions
        "sensitive": rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2]),  # protected group
        "score": rng.random(n),  # model scores (optional, e.g. probabilities)
    }
)

# Add mild bias: group "C" gets fewer positive predictions
mask_c = df["sensitive"] == "C"
df.loc[mask_c, "y_pred"] = (rng.random(mask_c.sum()) < 0.3).astype(int)

df.to_csv("dev_sample.csv", index=False)
print("✅ Created dev_sample.csv with", len(df), "rows")
