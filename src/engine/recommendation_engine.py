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

    def get_best_model(self):
        rule_engine = RuleEngine()
        results_dir = None
        
        for config_path in self.config_paths:
            logger.info(f"Running experiments from: {config_path}")
            model_results = rule_engine.run(config_path)
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
            if metrics_df['f1'].max() < self.threshold_f1 and metrics_df['latency'].max() < self.threshold_latency:
                return "Based on the evaluation results, fine-tuning does not provide significant performance improvements for the given datasets. We recommend proceeding with the RAG-based approach for better scalability and retrieval accuracy."
            if "latency" in self.factors and "accuracy" in self.factors:
                sorted_metrics_df = metrics_df.sort_values(by=['f1', 'latency'], ascending=[False, True])
            elif "latency" in self.factors:
                sorted_metrics_df = metrics_df.sort_values(by=['latency'], ascending=[True])
            elif "accuracy" in self.factors:
                sorted_metrics_df = metrics_df.sort_values(by=['f1'], ascending=[False])
            else:
                sorted_metrics_df = metrics_df
            return (
                f"Best Model for dataset '{dataset_name or 'All Datasets'}': "
                f"{sorted_metrics_df.iloc[0]['model']}/{sorted_metrics_df.iloc[0]['exp']} with F1 Score: {sorted_metrics_df.iloc[0]['f1']} "
                f"and Latency: {sorted_metrics_df.iloc[0]['latency']}"
            )
        except FileNotFoundError:
            return "Metrics file not found. Please ensure the path is correct and experiments have been saved."
        except Exception as e:
            return f"An error occurred while fetching the best model: {str(e)}"
