# Amazon ML Challenge 2026 Phase 1

Business entity resolution training/validation pipeline for the supplied Amazon ML Challenge dataset.

## Environment

Python 3.11+ recommended.

```powershell
pip install -r requirements.txt
```

The official dataset is not stored in Git. Expected local structure:

```text
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

## Phase 1 Command

```powershell
python -m src.pipeline --phase train --data-dir dataset
```

Phase 1 stops after train/validation work. It does not create final test predictions, final `matching_results.tsv`, final `candidate_pairs.tsv`, a submission zip, or any leaderboard submission.

## Pipeline

The pipeline verifies the four training files, runs full-dataset EDA, creates a fixed Source-1 entity-level train/validation split, normalizes names and addresses without overwriting originals, builds candidate pairs, engineers pair features, trains XGBoost, tunes thresholds, and evaluates entity-level macro F0.5.

Blocking uses these passes:

- exact conservative normalized business name
- suffix-normalized exact business name
- rare business-name token inverted index
- address numeric-token anchors
- sparse character 3-5 gram TF-IDF retrieval for business names
- sparse character 3-5 gram TF-IDF retrieval for addresses

Features include name/address equality, token overlap, fuzzy similarity via Python `SequenceMatcher`, TF-IDF retrieval scores/ranks, numeric/postal agreement, country equality, source indicator, missing flags, and blocking provenance.

Validation metric is entity-level macro F0.5 over Source-1 entities. Singleton truth with empty prediction scores 1.0; singleton truth with any prediction scores 0.0; non-singleton truth with empty prediction scores 0.0. Predictions use set semantics.

## Artifacts

Generated artifacts are written under `artifacts/`:

- `artifacts/eda/data_verification.json`
- `artifacts/eda/eda_summary.json`
- `artifacts/eda/ground_truth_examples.json`
- `artifacts/metrics/validation_split.json`
- `artifacts/metrics/baseline_exact_name_blocking.json`
- `artifacts/metrics/final_blocking_stats.json`
- `artifacts/metrics/candidate_pairs_phase1.pkl`
- `artifacts/metrics/threshold_tuning.csv`
- `artifacts/metrics/training_validation_metrics.json`
- `artifacts/metrics/validation_error_analysis.json`
- `artifacts/metrics/experiments.csv`
- `artifacts/metrics/phase1_execution_summary.json`
- `artifacts/models/xgboost_phase1.joblib`

## Known Limitations

The current implementation avoids external business data and internet enrichment. It uses installed local libraries only. If candidate recall is weak, improve blocking before Phase 2; if validation F0.5 is weak, inspect `validation_error_analysis.json` to separate blocking failures from feature/model/threshold failures.
