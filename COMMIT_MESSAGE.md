feat: Integrate fairness toolkit into recruitment model training script

Integrate comprehensive fairness assessment and mitigation capabilities
into the recruitment model training pipeline. The script now includes
bias detection, fairness-aware training, and post-training evaluation.

Changes:
- Add fairness toolkit imports (FairnessAnalyzer, detectors, transformers, Lagrangian trainer)
- Modify data loading to preserve sensitive attributes separately for fairness analysis
- Add pre-training bias detection (representation bias, statistical disparities, proxy variables)
- Add baseline fairness measurement using target distribution
- Add optional instance reweighting for data-level bias mitigation
- Replace standard training loop with LagrangianFairnessTrainer for fairness-constrained training
- Add post-training fairness evaluation (demographic parity, equalized odds)
- Add fairness threshold validation before model saving
- Fix import path resolution by adding project root to sys.path
- Fix data path to be relative to script location
- Fix detector result attribute access (pvalue, strength instead of effect_size, association)
- Add IDE configuration (.vscode/settings.json) for proper Python interpreter selection

The integrated pipeline now provides:
1. Pre-training bias detection on raw data
2. Baseline fairness metrics for comparison
3. Optional data transformation via instance reweighting
4. Fairness-aware training using Lagrangian method with demographic parity constraints
5. Post-training fairness evaluation with comparison to baseline
6. Fairness threshold validation (5% DP difference threshold)

Test subject files:
- train_recruit_model.py: Main training script with fairness integration
- generate_recruit_data.py: Data generation script
- synthetic_recruitment_data.csv: Training data
