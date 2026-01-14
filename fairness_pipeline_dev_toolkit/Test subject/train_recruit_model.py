# train_recruitment_model.py
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

# Add project root to Python path for imports BEFORE other imports
# File is at: fairness_pipeline_dev_toolkit/Test subject/train_recruit_model.py
# Need to go up 3 levels to reach project root
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from fairness_pipeline_dev_toolkit.integration.reporting import (  # noqa: E402
    generate_training_fairness_report,
)
from fairness_pipeline_dev_toolkit.metrics import FairnessAnalyzer  # noqa: E402
from fairness_pipeline_dev_toolkit.pipeline.detectors import (  # noqa: E402
    ProxyVariableDetector,
    RepresentationBiasDetector,
    StatisticalDisparityDetector,
)
from fairness_pipeline_dev_toolkit.pipeline.transformers.instance_reweighting import (  # noqa: E402
    InstanceReweighting,
)
from fairness_pipeline_dev_toolkit.training import (  # noqa: E402
    LagrangianFairnessTrainer,
)

# -----------------------------
# 1. Config
# -----------------------------
# Data path relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = str(SCRIPT_DIR / "synthetic_recruitment_data.csv")
TEST_SIZE = 0.2
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42

# Sensitive columns
SENSITIVE_COLS = ["race", "gender"]

# Numeric and categorical feature definitions
NUMERIC_COLS = [
    "years_experience",
    "num_roles",
    "skill_score",
    "interview_score",
]

CATEGORICAL_COLS = [
    "education_level",
    "previous_company_tier",
    "race",  # sensitive
    "gender",  # sensitive
]

TARGET_COL = "hired"


# -----------------------------
# 2. Dataset / Preprocessing
# -----------------------------
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess_data(
    path: str,
) -> Tuple[
    DataLoader, DataLoader, int, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Load CSV, preprocess, and return train/val dataloaders,
    input_dim, metadata for encoders/scaler, original dataframes, and sensitive attributes.

    Returns: train_loader, val_loader, input_dim, meta,
             train_df_orig, val_df_orig, sensitive_train_df, sensitive_val_df
    """
    df = pd.read_csv(path)

    assert TARGET_COL in df.columns, f"Target column {TARGET_COL} not found."

    # Train/validation split (stratified)
    train_df, val_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
    )

    # Store original dataframes for bias detection
    train_df_orig = train_df.copy()
    val_df_orig = val_df.copy()

    # Extract sensitive attributes (before encoding)
    sensitive_train_df = train_df[SENSITIVE_COLS].copy()
    sensitive_val_df = val_df[SENSITIVE_COLS].copy()

    # Fit transformers on train only
    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # Numeric features
    X_num_train = scaler.fit_transform(train_df[NUMERIC_COLS])
    X_num_val = scaler.transform(val_df[NUMERIC_COLS])

    # Categorical features
    X_cat_train = ohe.fit_transform(train_df[CATEGORICAL_COLS])
    X_cat_val = ohe.transform(val_df[CATEGORICAL_COLS])

    # Combine numeric + categorical
    X_train = np.hstack([X_num_train, X_cat_train])
    X_val = np.hstack([X_num_val, X_cat_val])

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]

    meta = {
        "scaler": scaler,
        "ohe": ohe,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }

    return (
        train_loader,
        val_loader,
        input_dim,
        meta,
        train_df_orig,
        val_df_orig,
        sensitive_train_df,
        sensitive_val_df,
    )


# -----------------------------
# 3. Model Definition
# -----------------------------
class RecruitmentNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 32), dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))  # binary classification

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 4. Training & Evaluation
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            running_loss += loss.item() * X_batch.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy()
            labels = y_batch.long().cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    val_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    return val_loss, acc, precision, recall, f1


# -----------------------------
# 5. Main Training Loop
# -----------------------------
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run generate_recruitment_data.py first.")

    print("Loading and preprocessing data...")
    (
        train_loader,
        val_loader,
        input_dim,
        meta,
        train_df_orig,
        val_df_orig,
        sensitive_train_df,
        sensitive_val_df,
    ) = load_and_preprocess_data(DATA_PATH)
    print(f"Train size: {meta['train_size']}, Val size: {meta['val_size']}")
    print(f"Input dimension: {input_dim}")
    print(f"Using device: {DEVICE}")

    print("\n" + "=" * 60)
    print("STEP 1: Bias Detection on Raw Data")
    print("=" * 60)

    # Initialize detectors
    rep_detector = RepresentationBiasDetector(alpha=0.05)
    disp_detector = StatisticalDisparityDetector(alpha=0.05)
    proxy_detector = ProxyVariableDetector(threshold=0.30)

    # Store detector results for report
    representation_bias_results = {}
    statistical_disparities_results = {}
    proxy_variables_results = {}

    # Run detection on training data
    print("\n--- Representation Bias ---")
    for attr in SENSITIVE_COLS:
        result = rep_detector.run(train_df_orig, attr, benchmark=None)
        representation_bias_results[attr] = result
        pval_str = f"{result.chi2_pvalue:.4f}" if result.chi2_pvalue is not None else "N/A"
        print(f"{attr}: flagged={result.flagged}, p-value={pval_str}")
        print(f"  Counts: {result.counts}")
        print(f"  Proportions: {result.proportions}")

    print("\n--- Statistical Disparities ---")
    for attr in SENSITIVE_COLS:
        disparities = disp_detector.run(train_df_orig, attr)
        statistical_disparities_results[attr] = disparities
        flagged_count = sum(1 for d in disparities if d.flagged)
        print(f"{attr}: {flagged_count} features flagged")
        for d in disparities[:3]:  # Show first 3
            print(f"  {d.feature}: test={d.test}, p-value={d.pvalue:.4f}, flagged={d.flagged}")

    print("\n--- Proxy Variables ---")
    for attr in SENSITIVE_COLS:
        proxies = proxy_detector.run(train_df_orig, attr)
        proxy_variables_results[attr] = proxies
        flagged_count = sum(1 for p in proxies if p.flagged)
        print(f"{attr}: {flagged_count} proxies flagged")
        for p in proxies[:3]:  # Show first 3
            print(f"  {p.feature}: {p.measure}={p.strength:.3f}, flagged={p.flagged}")

    print("\n" + "=" * 60)
    print("STEP 2: Baseline Fairness Measurement")
    print("=" * 60)

    # Initialize fairness analyzer
    analyzer = FairnessAnalyzer(min_group_size=30, backend="native")

    # For baseline, we'll use the target distribution as a proxy
    # In practice, you might train a simple baseline model first
    y_train_baseline = train_df_orig[TARGET_COL].values

    # Encode sensitive attributes for analysis (binary for demographic parity)
    # For multiple attributes, we'll analyze each separately
    baseline_metrics = {}

    for attr in SENSITIVE_COLS:
        # Convert to binary if needed (for demographic parity)
        # For multi-class, we'll use the first two groups or create binary splits
        s_train_attr = sensitive_train_df[attr].values

        # Simple binary encoding: use most common vs others, or first two groups
        unique_vals = pd.Series(s_train_attr).value_counts()
        if len(unique_vals) >= 2:
            # Use top 2 groups
            top_groups = unique_vals.head(2).index.tolist()
            s_binary = (pd.Series(s_train_attr).isin([top_groups[0]])).astype(int).values

            # Measure demographic parity on target distribution
            # (This is a proxy - in practice, use baseline model predictions)
            dp_result = analyzer.demographic_parity_difference(
                y_pred=y_train_baseline, sensitive=s_binary, with_ci=True, with_effect_size=True
            )
            baseline_metrics[f"{attr}_demographic_parity"] = dp_result
            print(f"\n{attr} - Demographic Parity Difference: {dp_result.value:.4f}")
            if dp_result.ci:
                print(f"  95% CI: [{dp_result.ci[0]:.4f}, {dp_result.ci[1]:.4f}]")

    print("\n" + "=" * 60)
    print("STEP 3: Optional Data Transformation (Instance Reweighting)")
    print("=" * 60)

    # Initialize reweighter
    reweighter = InstanceReweighting(
        sensitive=SENSITIVE_COLS, benchmarks=None, max_weight=10.0  # Use uniform balancing
    )

    # Fit on training data (returns weights, doesn't modify data)
    reweighter.fit(train_df_orig)
    sample_weights = reweighter.sample_weight_

    print(
        f"Sample weights computed: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}, mean={sample_weights.mean():.3f}"
    )

    # Store weights for use in training (if using weighted loss)
    # Note: PyTorch DataLoader can accept sample weights via WeightedRandomSampler

    # Prepare data for Lagrangian training
    # Convert DataLoader to full tensors (Lagrangian trainer expects full dataset)
    X_train_full = []
    y_train_full = []
    for X_batch, y_batch in train_loader:
        X_train_full.append(X_batch)
        y_train_full.append(y_batch)
    X_train_tensor = torch.cat(X_train_full, dim=0).to(DEVICE)
    y_train_tensor = torch.cat(y_train_full, dim=0).to(DEVICE).long().view(-1)

    # Prepare sensitive attributes for training
    # For Lagrangian trainer, we need binary encoding (0/1)
    # Strategy: Use first sensitive attribute (race) as primary, or create intersectional
    # For simplicity, we'll use race as binary (most common group vs others)
    s_train_race = sensitive_train_df[SENSITIVE_COLS[0]].values
    unique_races = pd.Series(s_train_race).value_counts()
    top_race = unique_races.index[0]
    s_train_binary = (pd.Series(s_train_race) == top_race).astype(int).values
    s_train_tensor = torch.tensor(s_train_binary, dtype=torch.long).to(DEVICE)

    # Initialize model
    model = RecruitmentNet(input_dim=input_dim).to(DEVICE)

    # Initialize Lagrangian trainer
    trainer = LagrangianFairnessTrainer(
        model=model,
        fairness="demographic_parity",  # or "equal_opportunity"
        dp_tolerance=0.02,  # Allow 2% difference in positive prediction rates
        eo_tolerance=0.02,
        model_lr=LR,
        lambda_lr=1e-2,  # Dual variable learning rate
        device=DEVICE,
    )

    print("\n" + "=" * 60)
    print("STEP 4: Fairness-Aware Training (Lagrangian Method)")
    print("=" * 60)

    # Train with fairness constraints
    training_history = trainer.fit(
        X_train_tensor,
        y_train_tensor,
        s_train_tensor,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True,
    )

    # Print training summary
    if training_history:
        final_epoch = training_history[-1]
        print("\nTraining Summary:")
        print(f"  Final Accuracy: {final_epoch['accuracy']:.4f}")
        print(f"  Final Violation: {final_epoch['violation']:.4f}")
        print(f"  Final Lambda: {final_epoch['lambda']:.4f}")

    # Best model is already in trainer.model
    best_state_dict = trainer.model.state_dict()

    print("\n" + "=" * 60)
    print("STEP 5: Post-Training Fairness Evaluation")
    print("=" * 60)

    # Get predictions on validation set
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy()
            val_predictions.append(preds)
            val_labels.append(y_batch.long().cpu().numpy())

    val_predictions = np.vstack(val_predictions).flatten()
    val_labels = np.vstack(val_labels).flatten()

    # Prepare sensitive attributes for validation (same encoding as training)
    s_val_race = sensitive_val_df[SENSITIVE_COLS[0]].values
    s_val_binary = (pd.Series(s_val_race) == top_race).astype(int).values

    # Compute fairness metrics
    final_metrics = {}

    # Demographic Parity
    dp_result = analyzer.demographic_parity_difference(
        y_pred=val_predictions, sensitive=s_val_binary, with_ci=True, with_effect_size=True
    )
    final_metrics["demographic_parity"] = dp_result

    # Equalized Odds (requires true labels)
    eo_result = analyzer.equalized_odds_difference(
        y_true=val_labels,
        y_pred=val_predictions,
        sensitive=s_val_binary,
        with_ci=True,
        with_effect_size=True,
    )
    final_metrics["equalized_odds"] = eo_result

    # Print results
    print("\nFinal Fairness Metrics (Validation Set):")
    print(f"  Demographic Parity Difference: {dp_result.value:.4f}")
    if dp_result.ci:
        print(f"    95% CI: [{dp_result.ci[0]:.4f}, {dp_result.ci[1]:.4f}]")
    print(f"  Equalized Odds Difference: {eo_result.value:.4f}")
    if eo_result.ci:
        print(f"    95% CI: [{eo_result.ci[0]:.4f}, {eo_result.ci[1]:.4f}]")

    # Compare to baseline
    if "race_demographic_parity" in baseline_metrics:
        baseline_dp = baseline_metrics["race_demographic_parity"].value
        improvement = baseline_dp - dp_result.value
        print(f"\nImprovement over baseline: {improvement:.4f}")
        print("  (Negative = reduction in unfairness)")

    # Fairness threshold check
    FAIRNESS_THRESHOLD = 0.05  # Maximum allowed demographic parity difference

    threshold_status = "pass" if dp_result.value <= FAIRNESS_THRESHOLD else "fail"

    if dp_result.value <= FAIRNESS_THRESHOLD:
        print(
            f"\n✓ Fairness threshold met: DP difference ({dp_result.value:.4f}) <= {FAIRNESS_THRESHOLD}"
        )
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            torch.save(model.state_dict(), "recruitment_model_best.pt")
            print("Model saved to recruitment_model_best.pt")
    else:
        print(
            f"\n⚠ Fairness threshold NOT met: DP difference ({dp_result.value:.4f}) > {FAIRNESS_THRESHOLD}"
        )
        print("Consider adjusting fairness constraints or mitigation strategies.")
        # Optionally still save, or raise an error
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            torch.save(model.state_dict(), "recruitment_model_best.pt")
            print("Model saved with warning.")

    # Generate comprehensive fairness report
    print("\n" + "=" * 60)
    print("Generating Comprehensive Fairness Report")
    print("=" * 60)

    # Prepare report data with raw inputs (function will auto-compute metrics)
    report_data = {
        "metadata": {
            "model_name": "RecruitmentNet",
            "sensitive_attributes": SENSITIVE_COLS,
            "fairness_threshold": FAIRNESS_THRESHOLD,
        },
        "data_stage": {
            "representation_bias": representation_bias_results,
            "statistical_disparities": statistical_disparities_results,
            "proxy_variables": proxy_variables_results,
        },
        "baseline_metrics": baseline_metrics,
        "mitigation": {
            "instance_reweighting": {
                "sample_weights": sample_weights,  # Pass raw weights, function will compute stats
            },
            "lagrangian_training": {
                "history": training_history,  # Pass raw history, function will compute convergence
            },
        },
        "final_metrics": final_metrics,
        "y_true": val_labels,  # Raw labels for auto-computing performance metrics
        "y_pred": val_predictions,  # Raw predictions for auto-computing performance metrics
        "comparison": {
            "improvement": improvement,
            "threshold_status": threshold_status,
        },
    }

    # Generate and save report
    artifacts_dir = project_root / "artifacts"
    markdown_report, json_data, file_paths = generate_training_fairness_report(
        report_data,
        output_dir=artifacts_dir,
    )

    if file_paths:
        print("\n✅ Comprehensive fairness report saved:")
        print(f"   - Markdown: {file_paths['markdown']}")
        print(f"   - JSON: {file_paths['json']}")


if __name__ == "__main__":
    main()
