"""Recommendation engine for model selection"""

import pandas as pd
import logging
import os

from .rule_engine import RuleEngine

logger = logging.getLogger(__name__)


class RecommendationEngine:
    def __init__(self, config_paths, threshold_f1=0.2, threshold_latency=0.3, factors=['accuracy', 'latency']):
        self.config_paths = config_paths
        self.threshold_f1 = threshold_f1
        self.threshold_latency = threshold_latency
        self.factors = factors

    def _sort_metrics(self, metrics_df):
        if "latency" in self.factors and "accuracy" in self.factors:
            return metrics_df.sort_values(by=['f1', 'latency'], ascending=[False, True])
        if "latency" in self.factors:
            return metrics_df.sort_values(by=['latency'], ascending=[True])
        if "accuracy" in self.factors:
            return metrics_df.sort_values(by=['f1'], ascending=[False])
        return metrics_df

    def _build_decision_summary(self, metrics_df, dataset_name):
        best_f1 = float(metrics_df['f1'].max())
        best_latency = float(metrics_df['latency'].min())

        baseline_df = metrics_df[metrics_df['exp'] == 'exp0']
        finetuned_df = metrics_df[metrics_df['exp'] != 'exp0']

        if best_f1 < self.threshold_f1 and best_latency < self.threshold_latency:
            return (
                "Decision: model is scalable via RAG (fine-tuning not recommended)",
                "Fine-tuning did not cross the configured quality/latency thresholds, so a scalable retrieval-first approach is preferred.",
            )

        if not baseline_df.empty and not finetuned_df.empty:
            baseline_best_f1 = float(baseline_df['f1'].max())
            finetuned_best_f1 = float(finetuned_df['f1'].max())
            if finetuned_best_f1 <= baseline_best_f1:
                return (
                    "Decision: model validation not increasing",
                    "Fine-tuned runs did not improve F1 over baseline (exp0), so validation gains are not increasing.",
                )

        return (
            "Decision: model is finetunable",
            f"Fine-tuning improved measurable quality for dataset '{dataset_name or 'All Datasets'}' under current thresholds.",
        )

    def get_best_model(self):
        rule_engine = RuleEngine()
        results_dir = None
        stop_reasons = []

        for config_path in self.config_paths:
            logger.info(f"Running experiments from: {config_path}")
            model_results = rule_engine.run(config_path)
            stop_reasons.extend(model_results.get("stop_reasons", []))
            results_dir = rule_engine.save_and_summarize_results(model_results)

        dataset_name = rule_engine.dataset_name
        dataset_label = (dataset_name or "dataset").replace(os.sep, "_").replace("/", "_")
        try:
            metrics_df = pd.read_csv(f"{results_dir}/metrics_{dataset_label}.csv")
            if metrics_df.empty:
                return "No metrics found for the experiments. Please run the experiments again."
            if dataset_name:
                metrics_df = metrics_df[metrics_df['dataset'] == dataset_name]
                if metrics_df.empty:
                    return f"No metrics found for dataset '{dataset_name}'. Please run the experiments for this dataset."

            sorted_metrics_df = self._sort_metrics(metrics_df)
            decision_title, decision_reason = self._build_decision_summary(metrics_df, dataset_name)
            best_row = sorted_metrics_df.iloc[0]

            stop_reason_text = ""
            if stop_reasons:
                stop_reason_text = "\nPipeline stop details:\n- " + "\n- ".join(stop_reasons)

            return (
                f"{decision_title}\n"
                f"Reason: {decision_reason}\n"
                f"Best Model for dataset '{dataset_name or 'All Datasets'}': "
                f"{best_row['model']}/{best_row['exp']} with F1 Score: {best_row['f1']} "
                f"and Latency: {best_row['latency']}"
                f"{stop_reason_text}"
            )
        except FileNotFoundError:
            return "Metrics file not found. Please ensure the path is correct and experiments have been saved."
        except Exception as e:
            return f"An error occurred while fetching the best model: {str(e)}"
