"""Rule-based experiment execution engine"""

import json
import operator
import os
import logging
from pandas.errors import ParserError

import pandas as pd

from ..data import DataPreparation
from ..models import FinetuneModel
from ..config_validation import validate_config
from ..reporting import generate_html_report

logger = logging.getLogger(__name__)


class RuleEngine:
    def __init__(self):
        self.system_prompt = None
        self.output_dir = None
        self.dataset_name = None

    def run_experiment(self, train_data, test_data, model_config, sft_config=None, mode="finetune"):
        logger.info("=" * 60)
        logger.info(f"Starting experiment for model: {model_config.get('model_name', 'unknown')}")
        logger.info(f"Chat template: {model_config.get('chat_template', 'unknown')}")
        if mode == "finetune":
            logger.info(
                "SFT config: batch_size=%s, epochs=%s, lr=%s",
                sft_config.get("batch_size"),
                sft_config.get("epochs"),
                sft_config.get("learning_rate"),
            )
        else:
            logger.info("Mode: baseline evaluation (no fine-tuning)")
        logger.info("=" * 60)
        finetuner = FinetuneModel(model_config, sft_config, self.system_prompt)
        model, tokenizer = finetuner.load_model()
        if mode == "finetune":
            model, tokenizer, logs = finetuner.train_model(model, tokenizer, train_data, test_data)
        else:
            logs = pd.DataFrame(columns=["step", "train_loss", "eval_loss"])
        results = finetuner.evaluate_model(model, tokenizer, test_data, model_config['chat_template'])
        logger.info("=" * 60)
        logger.info("Experiment completed.")
        logger.info("=" * 60)
        return model, tokenizer, logs, results

    def _extract_metric(self, results, key):
        exp_name, metric_name = key.split(".")
        if exp_name not in results['exp_results']:
            raise KeyError(
                f"Experiment '{exp_name}' has no results yet. "
                "Any rule depending on it cannot be evaluated right now."
            )
        exp_data = results['exp_results'][exp_name]
        if metric_name in exp_data["metrics"]:
            return exp_data["metrics"][metric_name]
        logs = exp_data["logs"]

        if logs.empty:
            raise KeyError(f"Logs are empty for experiment '{exp_name}'. Cannot extract metric '{metric_name}'.")

        if metric_name == "last_eval_loss":
            if "eval_loss" not in logs.columns:
                raise KeyError(f"Column 'eval_loss' not found in logs for experiment '{exp_name}'. Available columns: {list(logs.columns)}")
            eval_loss = logs["eval_loss"].dropna()
            if eval_loss.empty:
                raise KeyError(f"No non-null 'eval_loss' values found in logs for experiment '{exp_name}'.")
            return float(eval_loss.iloc[-1])
        if metric_name == "min_eval_loss":
            if "eval_loss" not in logs.columns:
                raise KeyError(f"Column 'eval_loss' not found in logs for experiment '{exp_name}'. Available columns: {list(logs.columns)}")
            eval_loss = logs["eval_loss"].dropna()
            if eval_loss.empty:
                raise KeyError(f"No non-null 'eval_loss' values found in logs for experiment '{exp_name}'.")
            return float(eval_loss.min())
        if metric_name == "last_train_loss":
            if "train_loss" not in logs.columns:
                raise KeyError(f"Column 'train_loss' not found in logs for experiment '{exp_name}'. Available columns: {list(logs.columns)}")
            train_loss = logs["train_loss"].dropna()
            if train_loss.empty:
                raise KeyError(f"No non-null 'train_loss' values found in logs for experiment '{exp_name}'.")
            return float(train_loss.iloc[-1])
        if metric_name == "min_train_loss":
            if "train_loss" not in logs.columns:
                raise KeyError(f"Column 'train_loss' not found in logs for experiment '{exp_name}'. Available columns: {list(logs.columns)}")
            train_loss = logs["train_loss"].dropna()
            if train_loss.empty:
                raise KeyError(f"No non-null 'train_loss' values found in logs for experiment '{exp_name}'.")
            return float(train_loss.min())
        raise KeyError(f"Unknown metric '{metric_name}' for {exp_name}")

    def _evaluate_rule_conditions(self, conditions, results):
        ops = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq, "!=": operator.ne}
        trace = []

        for cond in conditions:
            try:
                left_val = self._extract_metric(results, cond["left"])
                right_val = self._extract_metric(results, cond["right"])
            except KeyError as err:
                detail = {
                    "condition": cond,
                    "status": "unavailable",
                    "reason": str(err),
                }
                trace.append(detail)
                logger.warning("Skipping rule condition %s due to unavailable metric: %s", cond, err)
                return False, trace

            passed = ops[cond["op"]](left_val, right_val)
            detail = {
                "condition": cond,
                "status": "pass" if passed else "fail",
                "left_val": left_val,
                "right_val": right_val,
            }
            trace.append(detail)
            if not passed:
                return False, trace

        return True, trace

    def _check_conditions(self, conditions, results):
        passed, _ = self._evaluate_rule_conditions(conditions, results)
        return passed

    def _log_training_health(self, exp_name, logs):
        if logs is None or logs.empty:
            logger.info("[%s] Training logs unavailable; skipping overfit/loss trend diagnostics.", exp_name)
            return

        train_series = logs["train_loss"].dropna() if "train_loss" in logs.columns else pd.Series(dtype=float)
        eval_series = logs["eval_loss"].dropna() if "eval_loss" in logs.columns else pd.Series(dtype=float)

        if len(train_series) >= 2 and train_series.iloc[-1] > train_series.iloc[0]:
            logger.info("[%s] Training loss is increasing (start=%.6f, latest=%.6f).", exp_name, float(train_series.iloc[0]), float(train_series.iloc[-1]))
        elif len(train_series) >= 2:
            logger.info("[%s] Training loss trend is stable/decreasing (start=%.6f, latest=%.6f).", exp_name, float(train_series.iloc[0]), float(train_series.iloc[-1]))

        if len(eval_series) >= 2 and eval_series.iloc[-1] > eval_series.iloc[0]:
            logger.info("[%s] Validation loss is increasing (start=%.6f, latest=%.6f).", exp_name, float(eval_series.iloc[0]), float(eval_series.iloc[-1]))
        elif len(eval_series) >= 2:
            logger.info("[%s] Validation loss trend is stable/decreasing (start=%.6f, latest=%.6f).", exp_name, float(eval_series.iloc[0]), float(eval_series.iloc[-1]))

        if len(train_series) >= 2 and len(eval_series) >= 2:
            train_improved = train_series.iloc[-1] < train_series.iloc[0]
            eval_worsened = eval_series.iloc[-1] > eval_series.iloc[0]
            if train_improved and eval_worsened:
                logger.info("[%s] Model may be getting overfit (train loss down while validation loss up).", exp_name)

    def run(self, config_path):
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        validate_config(config, config_path)

        self.system_prompt = config['system_prompt']
        self.output_dir = config['output_dir']
        self.dataset_name = config['dataset']['name']

        data_loader = DataPreparation()
        experiments_data = data_loader.get_train_test_data(config['dataset'])
        test_data = experiments_data['test_data']
        experiments = config['experiments']
        auto_baseline = config.get("auto_baseline", True)

        result_config = {
            "model_name": experiments['exp1']['model']['model_name'],
            "chat_template": experiments['exp1']['model']['chat_template'],
            "output_dir": config['output_dir']
        }
        exp_results = {"config": result_config, "exp_results": {}, "stop_reasons": []}

        if auto_baseline:
            first_experiment = next(iter(experiments.values()))
            baseline_model_config = first_experiment["model"]
            logger.info("=" * 60)
            logger.info("Experiment: exp0 (automatic baseline)")
            logger.info("Train batch: none")
            logger.info("Run always: true")
            logger.info("Rules: none")
            logger.info("Reason: baseline always runs first for comparison")
            logger.info("=" * 60)
            logger.info("=" * 60)
            # Run baseline via direct evaluate-only path to avoid signature conflicts
            # across branches and keep baseline independent of training flow.
            finetuner = FinetuneModel(baseline_model_config, None, self.system_prompt)
            model, tokenizer = finetuner.load_model()
            logs = pd.DataFrame(columns=["step", "train_loss", "eval_loss"])
            metrics = finetuner.evaluate_model(model, tokenizer, test_data, baseline_model_config['chat_template'])
            exp_results['exp_results']["exp0"] = {
                "model": model,
                "tokenizer": tokenizer,
                "logs": logs,
                "metrics": metrics,
                "config": {
                    "mode": "baseline_eval",
                    "train_batch": None,
                    "model": baseline_model_config,
                    "sft": None,
                    "rules": [],
                },
            }

        for exp, exp_config in experiments.items():
            train_data = experiments_data[exp_config['train_batch']]
            model_config = exp_config['model']
            sft_config = exp_config['sft']
            logger.info("=" * 60)
            logger.info("Experiment: %s", exp)
            logger.info("Train batch: %s", exp_config['train_batch'])
            logger.info("Run always: %s", exp_config.get('run_always', False))
            logger.info("Rules: %s", "none" if not exp_config['rules'] else exp_config['rules'])
            logger.info("=" * 60)

            if not exp_config['rules']:
                logger.info("[CONTINUE] %s has no rules, so it will run.", exp)
                model, tokenizer, logs, metrics = self.run_experiment(train_data, test_data, model_config, sft_config)
                self._log_training_health(exp, logs)
                result = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "logs": logs,
                    "metrics": metrics,
                    "config": {
                        "train_batch": exp_config["train_batch"],
                        "model": model_config,
                        "sft": sft_config,
                        "rules": exp_config["rules"],
                    },
                }
                exp_results['exp_results'][exp] = result
                continue

            experiment_ran = False
            last_rule_failure_reason = None
            for rule_idx, rule in enumerate(exp_config['rules'], start=1):
                passed, trace = self._evaluate_rule_conditions(rule['conditions'], exp_results)
                for condition_trace in trace:
                    condition = condition_trace["condition"]
                    if condition_trace["status"] == "unavailable":
                        logger.info(
                            "[RULE %s - %s] UNAVAILABLE -> %s %s %s (%s)",
                            rule_idx,
                            exp,
                            condition['left'],
                            condition['op'],
                            condition['right'],
                            condition_trace['reason'],
                        )
                    else:
                        logger.info(
                            "[RULE %s - %s] %s -> %s %s %s (%.6f %s %.6f)",
                            rule_idx,
                            exp,
                            condition_trace['status'].upper(),
                            condition['left'],
                            condition['op'],
                            condition['right'],
                            float(condition_trace['left_val']),
                            condition['op'],
                            float(condition_trace['right_val']),
                        )

                if not passed:
                    if trace:
                        last_trace = trace[-1]
                        condition = last_trace["condition"]
                        if last_trace["status"] == "unavailable":
                            last_rule_failure_reason = (
                                f"rule {rule_idx} unavailable because {condition['left']} {condition['op']} {condition['right']}"
                                f" could not be evaluated ({last_trace['reason']})"
                            )
                        else:
                            last_rule_failure_reason = (
                                f"rule {rule_idx} failed because {condition['left']} {condition['op']} {condition['right']}"
                                f" evaluated as {float(last_trace['left_val']):.6f} {condition['op']} {float(last_trace['right_val']):.6f}"
                            )
                    logger.info("[STOP] %s rule %s failed; checking next rule if available.", exp, rule_idx)
                    continue

                logger.info("[CONTINUE] %s rule %s passed, running experiment.", exp, rule_idx)
                model, tokenizer, logs, metrics = self.run_experiment(train_data, test_data, model_config, sft_config)
                self._log_training_health(exp, logs)
                result = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "logs": logs,
                    "metrics": metrics,
                    "config": {
                        "train_batch": exp_config["train_batch"],
                        "model": model_config,
                        "sft": sft_config,
                        "rules": exp_config["rules"],
                    },
                }
                exp_results['exp_results'][exp] = result
                experiment_ran = True
                if not exp_config['run_always']:
                    stop_reason = f"{exp} stopped pipeline because run_always is false after successful run."
                    exp_results["stop_reasons"].append(stop_reason)
                    logger.info("[STOP] %s", stop_reason)
                    return exp_results

            if not experiment_ran:
                if last_rule_failure_reason:
                    stop_reason = f"{exp} did not run because no rule passed; last failure: {last_rule_failure_reason}."
                else:
                    stop_reason = f"{exp} did not run because no rule passed."
                exp_results["stop_reasons"].append(stop_reason)
                logger.info("[STOP] %s", stop_reason)
        return exp_results

    def save_and_summarize_results(self, results):
        try:
            from datetime import datetime
            exp_results = results["exp_results"]
            config = results["config"]
            model_name = config["model_name"]
            output_dir = config['output_dir']
            dataset_label = (self.dataset_name or "dataset").replace(os.sep, "_").replace("/", "_")
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
                log_path = os.path.join(output_dir, f"logs_{dataset_label}_{exp_name}.csv")
                logs.to_csv(log_path, index=False)
                model.save_pretrained(exp_dir)
                tokenizer.save_pretrained(exp_dir)
                metrics_list.append({**metrics, "exp": exp_name, "model": model_name, "dataset": self.dataset_name, "date": current_date})
                f1 = metrics.get("f1", None)
                if f1 is not None and f1 > max_f1:
                    max_f1 = f1
                    best_model = f"{model_name}/{exp_name}"

            metrics_path = os.path.join(output_dir, f"metrics_{dataset_label}.csv")
            metrics_df = pd.DataFrame(metrics_list)
            if os.path.exists(metrics_path):
                try:
                    existing_metrics_df = pd.read_csv(metrics_path)
                except ParserError:
                    logger.warning(
                        "Existing metrics file is malformed and will be repaired: %s",
                        metrics_path,
                    )
                    existing_metrics_df = pd.read_csv(metrics_path, on_bad_lines="skip")
                metrics_df = pd.concat([existing_metrics_df, metrics_df], ignore_index=True, sort=False)
            metrics_df.to_csv(metrics_path, index=False)
            try:
                report_html = generate_html_report(
                    results=results,
                    output_dir=output_dir,
                    dataset_name=self.dataset_name or "dataset",
                    dataset_label=dataset_label,
                )
                report_path = os.path.join(output_dir, f"report_{dataset_label}.html")
                with open(report_path, "w", encoding="utf-8") as report_file:
                    report_file.write(report_html)
                logger.info(f"HTML report written to {report_path}")
            except Exception as report_error:
                logger.warning(f"Failed to generate HTML report: {report_error}")
            logger.info(f"Results saved successfully. Best model: {best_model} (F1={max_f1:.4f})")
            return output_dir
        except Exception as e:
            logger.error(f"Error while saving and summarizing results: {str(e)}", exc_info=True)
