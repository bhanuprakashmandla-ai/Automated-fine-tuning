"""Configuration validation helpers."""

from __future__ import annotations

from typing import Any, Dict


def _require_keys(obj: Dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {', '.join(missing)}")


def _require_non_empty(value: Any, context: str) -> None:
    if value is None or value == "" or value == [] or value == {}:
        raise ValueError(f"{context} must not be empty.")


def validate_config(config: Dict[str, Any], config_path: str | None = None) -> None:
    """Validate experiment configuration structure.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    prefix = f"{config_path}: " if config_path else ""
    _require_keys(config, ["dataset", "output_dir", "system_prompt", "experiments"], "root")
    _require_non_empty(config.get("output_dir"), f"{prefix}output_dir")
    _require_non_empty(config.get("system_prompt"), f"{prefix}system_prompt")

    dataset = config["dataset"]
    _require_keys(dataset, ["type", "input_fields", "output_fields", "batch_config"], "dataset")
    _require_non_empty(dataset.get("type"), f"{prefix}dataset.type")
    _require_non_empty(dataset.get("input_fields"), f"{prefix}dataset.input_fields")
    _require_non_empty(dataset.get("output_fields"), f"{prefix}dataset.output_fields")

    batch_config = dataset["batch_config"]
    _require_keys(
        batch_config,
        ["first_batch", "second_batch", "third_batch", "test_batch"],
        "dataset.batch_config",
    )

    dataset_type = dataset.get("type")
    splitter = dataset.get("splitter")
    if dataset_type == "huggingface":
        _require_non_empty(dataset.get("path"), f"{prefix}dataset.path")
    elif splitter == "pdf":
        _require_non_empty(dataset.get("path"), f"{prefix}dataset.path")
        _require_keys(dataset, ["pdf_config"], "dataset")
        pdf_config = dataset["pdf_config"]
        _require_keys(
            pdf_config,
            ["llm_config", "chunk_size", "overlap", "qa_pairs_per_chunk", "max_generation_tokens"],
            "dataset.pdf_config",
        )
        llm_config = pdf_config["llm_config"]
        _require_keys(llm_config, ["api_base", "model_name"], "dataset.pdf_config.llm_config")
    elif dataset_type == "file":
        _require_non_empty(dataset.get("path"), f"{prefix}dataset.path")
        _require_non_empty(dataset.get("splitter"), f"{prefix}dataset.splitter")
    else:
        raise ValueError(
            f"{prefix}dataset.type must be 'huggingface' or 'file', or "
            "dataset.splitter must be 'pdf'."
        )

    experiments = config["experiments"]
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError(f"{prefix}experiments must be a non-empty object.")

    for exp_name, exp_config in experiments.items():
        context = f"experiments.{exp_name}"
        _require_keys(exp_config, ["run_always", "train_batch", "model", "sft", "rules"], context)
        _require_non_empty(exp_config.get("train_batch"), f"{prefix}{context}.train_batch")

        model_config = exp_config["model"]
        _require_keys(
            model_config,
            ["model_name", "chat_template", "max_seq_len", "rank", "alpha", "dropout"],
            f"{context}.model",
        )

        if "eval_batch_size" in model_config:
            eval_batch_size = model_config["eval_batch_size"]
            if not isinstance(eval_batch_size, int) or eval_batch_size < 1:
                raise ValueError(
                    f"{prefix}{context}.model.eval_batch_size must be an integer >= 1 when provided."
                )

        sft_config = exp_config["sft"]
        _require_keys(
            sft_config,
            ["batch_size", "epochs", "learning_rate", "logging_steps", "eval_steps", "save_steps", "eval_accumulation_steps"],
            f"{context}.sft",
        )
