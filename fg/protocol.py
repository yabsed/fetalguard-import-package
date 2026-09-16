"""Versioned, predeclared comparisons. Written before any model is fitted."""
from .common import write_json

DESIGN_VERSION = "2.0"


def protocol(cfg):
    return {
        "design_version": DESIGN_VERSION,
        "profile": cfg["profile"],
        "role": "execution_test_only" if cfg["profile"] == "mock" else "prespecified_evaluation",
        "primary_target": "expert_segment_abnormality",
        "prediction_unit": "five_minute_selected_fetus_segment",
        "split_and_bootstrap_unit": "mother",
        "cohort_history": "supplied" if cfg.get("prior_cohort_file") else "not_checked",
        "primary_metric": "auprc",
        "secondary_metrics": ["auroc", "sensitivity", "specificity", "ppv", "brier", "normal_record_alarm_rate"],
        "selection": {"trees": "validation_AP_best_seed", "single_features": "validation_AUROC",
                      "cnn": "width_by_mean_validation_AP_then_best_validation_seed",
                      "threshold": "validation_negative_90th_percentile_conservative_ties"},
        "hypotheses": {
            "H1": {"question": "How much of expert reading can waveform factors predict?",
                   "comparisons": ["cat28", "xgb28", "cat18", "logistic28"],
                   "interpretation": "AP baseline is prevalence; published results on different cohorts are context only."},
            "H2": {"question": "Does a combination beat a flexible single-factor predictor?",
                   "primary_metric": "auroc", "secondary_metric": "auprc",
                   "comparisons": ["best_single", "best_single_nonlinear", "logistic28", "cat28"],
                   "interpretation": "The nonlinear single predictor is required before inferring a need for multiple factors."},
            "H3": {"question": "Which groups contribute and is that robust to extraction?",
                   "comparisons": ["six_group_ablations", "robust", "cat18", "cat28_unsmoothed"],
                   "interpretation": "Measurement quality precedes interpretation. Predictive contribution is not causality."},
            "H4": {"question": "What is the performance and cost tradeoff against small CNNs?",
                   "comparisons": ["cat28_vs_selected_CNN", "normalization_same_width_and_seed"],
                   "interpretation": "No equivalence/noninferiority declaration. No prespecified justified margin exists."},
            "H5": {"question": "How does expert reading associate with outcomes and add to clinical predictors?",
                   "comparisons": ["reading_outcome_association", "clinical_plus_reading_minus_clinical"],
                   "interpretation": "Exploratory; observed outcomes only. Nonsignificant increment is not absence of association."},
        },
        "sensitivity": {"feature_smoothing_seconds": [30, 0], "primary_smoothing_seconds": 30,
                        "univariate_spline_knots": 5,
                        "same_patients_and_training_budget": True},
        "limitations": ["Site codes are provenance groups, not verified hospitals.",
            "Classical sample-to-sample STV is not clinical beat-to-beat STV.",
            "Duration >5 minutes cannot be measured within a 5-minute window.",
            "Cluster intervals reflect evaluation sampling, not the entire model-selection process.",
            "Prior-cohort exclusion applies to the primary holdout; CV, H5 and LOSO remain exploratory on the available cohort.",
            "Multiple secondary comparisons are exploratory; no familywise significance claim.",
            "Official models have a different target/subset and unknown pretraining overlap."],
        "budget": cfg["budget"],
        "export_policy": {"status": "pending_institution_review", "min_mothers_screen": cfg.get("export_min_mothers", 10)},
    }


def write_protocol(run, cfg):
    write_json(run / "protocol.json", protocol(cfg))
