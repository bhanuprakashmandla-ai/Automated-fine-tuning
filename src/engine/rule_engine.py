"""Rule-based experiment execution engine"""

import json
import operator
import os
import logging

import pandas as pd

from ..data import DataPreparation
from ..models import FinetuneModel

logger = logging.getLogger(__name__)


class RuleEngine:
    def __init__(self):
        self.system_prompt = None
        self.output_dir = None
        self.dataset_name = None

    def run_experiment(self, train_data, test_data, model_config, sft_config):
        finetuner = FinetuneModel(model_config, sft_config, self.system_prompt)
        model, tokenizer = finetuner.load_model()
        model, tokenizer, logs = finetuner.train_model(model, tokenizer, train_data, test_data)
        results = finetuner.evaluate_model(model, tokenizer, test_data, model_config['chat_template'])
        return model, tokenizer, logs, results

    def _check_conditions(self, conditions, results):
        OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq, "!=": operator.ne}
        
        def get_metric(results, key):
            exp_name, metric_name = key.split(".")
            exp_data = results['exp_results'][exp_name]
            if metric_name in exp_data["metrics"]:
                return exp_data["metrics"][metric_name]
            logs = exp_data["logs"]
            
            # Check if logs DataFrame is empty
            if logs.empty:
                raise KeyError(f"Logs are empty for experiment '{exp_name}'. Cannot extract metric '{metric_name}'.")
            
            if metric_name == "last_eval_loss":
                if "eval_loss" not in logs.columns:
                    raise KeyError(f"Column 'eval_loss' not found in logs for experiment '{exp_name}'. Available columns: {list(logs.columns)}")
                return float(logs["eval_loss"].dropna().iloc[-1])
            if metric_name == "min_eval_loss":
                if "eval_loss" not in logs.columns:
                    raise KeyError(f"Column 'eval_loss' not found in logs for experiment '{exp_name}'. Available columns: {list(logs.columns)}")
                return float(logs["eval_loss"].dropna().min())
            if metric_name == "last_train_loss":
                if "train_loss" not in logs.columns:
                    raise KeyError(f"Column 'train_loss' not found in logs for experiment '{exp_name}'. Available columns: {list(logs.columns)}")
                return float(logs["train_loss"].dropna().iloc[-1])
            if metric_name == "min_train_loss":
                if "train_loss" not in logs.columns:
                    raise KeyError(f"Column 'train_loss' not found in logs for experiment '{exp_name}'. Available columns: {list(logs.columns)}")
                return float(logs["train_loss"].dropna().min())
            raise KeyError(f"Unknown metric '{metric_name}' for {exp_name}")

        for cond in conditions:
            left_val = get_metric(results, cond["left"])
            right_val = get_metric(results, cond["right"])
            op_func = OPS[cond["op"]]
            if not op_func(left_val, right_val):
                return False
        return True

    def run(self, config_path):
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        self.system_prompt = config['system_prompt']
        self.output_dir = config['output_dir']
        self.dataset_name = config['dataset']['name']

        data_loader = DataPreparation()
        experiments_data = data_loader.get_train_test_data(config['dataset'])
        test_data = experiments_data['test_data']
        experiments = config['experiments']
        
        result_config = {
            "model_name": experiments['exp1']['model']['model_name'],
            "chat_template": experiments['exp1']['model']['chat_template'],
            "output_dir": config['output_dir']
        }
        exp_results = {"config": result_config, "exp_results": {}}

        for exp, config in experiments.items():
            train_data = experiments_data[config['train_batch']]
            model_config = config['model']
            sft_config = config['sft']
            if not config['rules']:
                model, tokenizer, logs, metrics = self.run_experiment(train_data, test_data, model_config, sft_config)
                result = {"model": model, "tokenizer": tokenizer, "logs": logs, "metrics": metrics}
                exp_results['exp_results'][exp] = result
            else:
                for rule in config['rules']:
                    if self._check_conditions(rule['conditions'], exp_results):
                        model, tokenizer, logs, metrics = self.run_experiment(train_data, test_data, model_config, sft_config)
                        result = {"model": model, "tokenizer": tokenizer, "logs": logs, "metrics": metrics}
                        exp_results['exp_results'][exp] = result
                        if not config['run_always']:
                            return exp_results
        return exp_results

    def save_and_summarize_results(self, results):
        try:
            from datetime import datetime
            exp_results = results["exp_results"]
            config = results["config"]
            model_name = config["model_name"]
            output_dir = config['output_dir']
            metrics_list = []
            max_f1 = float("-inf")
            best_model = None
            current_date = datetime.now().strftime("%Y-%m-%d")
            os.makedirs(output_dir, exist_ok=True)

            for exp_name, exp_data in exp_results.items():
                logs = exp_data["logs"]
                metrics = exp_data["metrics"]
                model = exp_data["model"]
                tokenizer = exp_data["tokenizer"]
                exp_dir = os.path.join(output_dir, "models", model_name, exp_name)
                os.makedirs(exp_dir, exist_ok=True)
                log_path = os.path.join(output_dir, f"logs_{self.dataset_name}_{exp_name}.csv")
                logs.to_csv(log_path, index=False)
                model.save_pretrained(exp_dir)
                tokenizer.save_pretrained(exp_dir)
                metrics_list.append({**metrics, "exp": exp_name, "model": model_name, "dataset": self.dataset_name, "date": current_date})
                f1 = metrics.get("f1", None)
                if f1 is not None and f1 > max_f1:
                    max_f1 = f1
                    best_model = f"{model_name}/{exp_name}"

            metrics_path = os.path.join(output_dir, f"metrics_{self.dataset_name}.csv")
            metrics_df = pd.DataFrame(metrics_list)
            if os.path.exists(metrics_path):
                metrics_df.to_csv(metrics_path, mode="a", index=False, header=False)
            else:
                metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Results saved successfully. Best model: {best_model} (F1={max_f1:.4f})")
            return output_dir
        except Exception as e:
            logger.error(f"Error while saving and summarizing results: {str(e)}", exc_info=True)
