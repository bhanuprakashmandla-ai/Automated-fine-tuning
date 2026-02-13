"""Model fine-tuning and evaluation"""

import os
import platform
import time
import logging

import pandas as pd
import torch

# Import torch._dynamo at module level to avoid scoping issues
try:
    import torch._dynamo
except Exception:
    torch._dynamo = None

# OS-specific configuration
IS_WINDOWS = platform.system() == "Windows"

# Set environment variables BEFORE any imports to ensure they're loaded
if IS_WINDOWS:
    os.environ["UNSLOTH_ENABLE_CCE"] = "0"  # Disable optimized cross-entropy loss
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  # Disable torch.compile entirely
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Memory optimization

# Disable torch.compile globally on Windows to avoid Triton compilation errors
if IS_WINDOWS and torch._dynamo is not None:
    try:
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True
    except Exception:
        pass

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from sentence_transformers import SentenceTransformer, util
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

logger = logging.getLogger(__name__)


class FinetuneModel:
    def __init__(self, model_config, sft_config, system_prompt):
        self.model_config = model_config
        self.sft_config = sft_config
        self.system_prompt = system_prompt

    def load_model(self):
        logger.info(f"Loading model: {self.model_config['model_name']}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=self.model_config['model_name'],
            max_seq_length=self.model_config['max_seq_len'],
            load_in_4bit=True,
        )

        model = FastModel.get_peft_model(
            model,
            r=self.model_config['rank'],
            target_modules=[
                "q_proj", "k_proj", "v_proj",
                "o_proj", "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=self.model_config['alpha'],
            lora_dropout=self.model_config['dropout'],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=357841,
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template=self.model_config['chat_template']
        )

        return model, tokenizer

    def train_model(self, model, tokenizer, train_data, test_data):
        logger.info("Starting training...")
        
        # Clear CUDA cache before training to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Get GPU memory info
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU Memory: {gpu_memory_gb:.2f} GB")
            
            # For GPUs with < 6GB, reduce batch size and increase gradient accumulation
            if gpu_memory_gb < 6.0:
                original_batch_size = self.sft_config["batch_size"]
                if original_batch_size > 4:
                    # Reduce batch size and increase gradient accumulation to maintain effective batch size
                    self.sft_config["batch_size"] = min(4, original_batch_size // 2)
                    gradient_accumulation = max(2, original_batch_size // self.sft_config["batch_size"])
                    logger.info(f"Reduced batch size from {original_batch_size} to {self.sft_config['batch_size']} "
                              f"and increased gradient accumulation to {gradient_accumulation} for low-memory GPU")
                else:
                    gradient_accumulation = 1
            else:
                gradient_accumulation = 1
        else:
            gradient_accumulation = 1
        
        # OS-specific fixes for multiprocessing and Triton issues
        if IS_WINDOWS:
            os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
            
            # Monkey-patch datasets to force single-process mode on Windows
            # This prevents subprocess spawning that can't import UnslothSFTTrainer
            try:
                from datasets.arrow_dataset import Dataset as ArrowDataset
                original_map = ArrowDataset.map
                def patched_map(self, *args, **kwargs):
                    if "num_proc" in kwargs and kwargs["num_proc"] is not None:
                        kwargs["num_proc"] = None
                    return original_map(self, *args, **kwargs)
                ArrowDataset.map = patched_map
                logger.info("Patched datasets.ArrowDataset.map for Windows compatibility")
            except Exception as e:
                logger.warning(f"Could not patch datasets: {e}")

        supports_bf16 = torch.cuda.is_bf16_supported()

        def process_batch(examples):
            texts = []
            for q, a in zip(examples["question"], examples["answer"]):
                conversation = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": str(q)},
                    {"role": "assistant", "content": str(a)},
                ]
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if "gemma" in self.model_config["chat_template"].lower():
                    text = text.removeprefix("<bos>")
                texts.append(text)
            return {"text": texts}

        # OS-safe multiprocessing (Windows uses single process, others use all CPUs)
        num_proc = 1 if IS_WINDOWS else os.cpu_count()

        train_dataset = train_data.map(
            process_batch,
            batched=True,
            num_proc=num_proc,
            remove_columns=train_data.column_names,
            desc="Formatting Train Data",
        )

        test_dataset = test_data.map(
            process_batch,
            batched=True,
            num_proc=num_proc,
            remove_columns=test_data.column_names,
            desc="Formatting Test Data",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            args=SFTConfig(
                dataset_text_field="text",
                dataset_num_proc=1 if IS_WINDOWS else None,  # Force single-process on Windows
                per_device_train_batch_size=self.sft_config["batch_size"],
                gradient_accumulation_steps=gradient_accumulation,
                warmup_steps=int(0.03 * len(train_dataset)),
                num_train_epochs=self.sft_config["epochs"],
                learning_rate=self.sft_config["learning_rate"],
                logging_steps=self.sft_config["logging_steps"],
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=357841,
                output_dir="output",
                report_to="none",
                eval_accumulation_steps=self.sft_config["eval_accumulation_steps"],
                save_strategy="steps",
                save_steps=self.sft_config["save_steps"],
                save_total_limit=5,
                eval_strategy="steps",  # Changed from evaluation_strategy to eval_strategy
                eval_steps=self.sft_config["eval_steps"],
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=not supports_bf16,
                bf16=supports_bf16,
            ),
        )

        if self.sft_config.get("early_stopping_criteria"):
            trainer.add_callback(
                EarlyStoppingCallback(
                    early_stopping_patience=5,
                    early_stopping_threshold=0.0,
                )
            )

        if "gemma" in self.model_config["chat_template"].lower():
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<start_of_turn>user\n",
                response_part="<start_of_turn>model\n",
            )
        elif "qwen" in self.model_config["chat_template"].lower():
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )

        trainer.train()

        logs = self.format_logs(trainer.state.log_history)
        return model, tokenizer, logs

    def evaluate_model(self, model, tokenizer, test_data, template):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()

        results = []
        eval_batch_size = int(self.model_config.get("eval_batch_size", 4) or 4)
        eval_batch_size = max(eval_batch_size, 1)
        rows = list(test_data)
        if not rows:
            logger.warning("No rows available in test_data during evaluation.")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "latency": 0.0, "semantic_similarity": 0.0}

        logger.info("Evaluating %s samples with eval_batch_size=%s", len(rows), eval_batch_size)

        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        for start_idx in range(0, len(rows), eval_batch_size):
            batch_rows = rows[start_idx:start_idx + eval_batch_size]
            batch_start_time = time.time()

            batch_texts = []
            references = []
            for row in batch_rows:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": row["question"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                batch_texts.append(text)
                references.append(str(row["answer"]))

            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)

            # Keep evaluation deterministic across all experiments.
            # Hard-code greedy decoding so checkpoint generation_config values
            # (e.g., do_sample=True) cannot change evaluation behavior.
            generation_kwargs = {
                "max_new_tokens": 125,
                "do_sample": False,
            }

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            predictions = []
            for idx in range(len(batch_rows)):
                prompt_len = int(inputs["attention_mask"][idx].sum().item())
                prediction = tokenizer.decode(
                    outputs[idx][prompt_len:],
                    skip_special_tokens=True,
                ).strip()
                predictions.append(prediction)

            prediction_embeddings = embed_model.encode(predictions, convert_to_tensor=True)
            reference_embeddings = embed_model.encode(references, convert_to_tensor=True)
            similarities = util.cos_sim(prediction_embeddings, reference_embeddings).diagonal().tolist()

            batch_latency_per_sample = (time.time() - batch_start_time) / len(batch_rows)

            for prediction, reference, similarity in zip(predictions, references, similarities):
                precision, recall, f1 = self.token_f1(prediction, reference)
                results.append(
                    {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "latency": batch_latency_per_sample,
                        "semantic_similarity": float(similarity),
                    }
                )

        return pd.DataFrame(results).mean().to_dict()

    def token_f1(self, prediction, reference):
        p = prediction.lower().split()
        r = reference.lower().split()
        common = set(p) & set(r)
        if not common:
            return 0.0, 0.0, 0.0
        precision = len(common) / len(p)
        recall = len(common) / len(r)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    def format_logs(self, logs):
        train = [l for l in logs if "loss" in l]
        evals = [l for l in logs if "eval_loss" in l]

        df_train = pd.DataFrame(train)[["step", "loss"]] if train else pd.DataFrame()
        df_eval = pd.DataFrame(evals)[["step", "eval_loss"]] if evals else pd.DataFrame()

        # Rename "loss" to "train_loss" to match rule engine expectations
        if not df_train.empty:
            df_train = df_train.rename(columns={"loss": "train_loss"})

        if not df_train.empty and not df_eval.empty:
            return pd.merge(df_train, df_eval, on="step", how="outer")
        return df_train
