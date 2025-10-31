# Pipeline Run Report

- **Config**: `fairness_pipeline_dev_toolkit/pipeline/pipeline.config.yml`
- **Input CSV**: `dev_sample.csv`
- **Output CSV**: `artifacts/sample.transformed.csv`

## Detector Findings (summary)
- **meta**: 3 entries
- **summary**: 6 entries
- **representation**: [{'attribute': 'sensitive', 'counts': {'A': 98, 'B': 62, 'C': 40}, 'proportions': {'A': 0.49, 'B': 0.31, 'C': 0.2}, 'benchmark': {'A': 0.4, 'B': 0.4, 'C': 0.2}, 'chi2_pvalue': 0.12861170791754997, 'flagged': False}]
- **disparities**: [{'feature': 'y_true', 'attribute': 'sensitive', 'test': 'chi2', 'pvalue': 0.3569255347473493, 'flagged': False}, {'feature': 'y_pred', 'attribute': 'sensitive', 'test': 'chi2', 'pvalue': 0.009580363168086106, 'flagged': True}, {'feature': 'score', 'attribute': 'sensitive', 'test': 'anova', 'pvalue': 0.13902782607515096, 'flagged': False}]
- **proxies**: [{'feature': 'y_true', 'attribute': 'sensitive', 'measure': 'cramers_v', 'strength': 0.10150015296054936, 'flagged': False}, {'feature': 'y_pred', 'attribute': 'sensitive', 'measure': 'cramers_v', 'strength': 0.2155931301950939, 'flagged': False}, {'feature': 'score', 'attribute': 'sensitive', 'measure': 'eta_squared', 'strength': 0.019831987806349694, 'flagged': False}]