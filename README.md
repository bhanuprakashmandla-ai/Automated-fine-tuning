# Automated Fine-tuning System

A modular, production-ready system for automated LLM fine-tuning with rule-based experiment execution and intelligent model selection. Cross-platform compatible (Windows, Linux) with automatic GPU detection and CUDA support.

## 🌟 Features

- **🚀 Automated Fine-tuning**: Seamless integration with Unsloth for efficient model training
- **📊 Rule-based Execution**: Smart experiment orchestration based on custom conditions
- **🎯 Multi-model Support**: Compatible with Gemma, Qwen, and other popular models
- **📈 Comprehensive Metrics**: Precision, Recall, F1 Score, and Latency tracking
- **🔄 PDF-to-QA Generation**: Automatic dataset creation from PDF documents
- **💾 Dataset-specific Results**: Organized output with date tracking per dataset
- **🌐 Cross-platform**: Works on Windows, Linux, Google Colab, and Cloud environments
- **📝 Advanced Logging**: Detailed logging system for debugging and monitoring
- **🖥️ GPU Optimization**: Intelligent GPU detection with automatic CUDA-compatible PyTorch installation
- **💾 Memory Management**: Automatic batch size adjustment for low-memory GPUs

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Platform-Specific Notes](#platform-specific-notes)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Prerequisites

- **Python 3.8 or higher**
- **NVIDIA GPU** (recommended, 4GB+ VRAM) - Works with any NVIDIA GPU (RTX, GTX, Quadro, etc.)
- **NVIDIA GPU Driver** (required for GPU acceleration)
  - Verify driver installation: `nvidia-smi` in terminal
  - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- **16GB+ RAM** (recommended)
- **Internet connection** (for downloading models and packages)

## 🔧 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/FloTorch/Automated-fine-tuning.git
cd Automated-fine-tuning
```

### Step 2: Create Virtual Environment

**On Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

**On Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**Note:** After activation, your terminal prompt should show `(venv)` at the beginning.

### Step 3: Install Packages

The system includes an automated package installer that detects your OS and GPU, then installs the appropriate packages:

```bash
# Install all required packages (automatic GPU detection)
python run.py --install-packages
```

**What this command does:**
- ✅ Upgrades pip to the latest version
- ✅ Detects your operating system (Windows/Linux)
- ✅ Installs Unsloth (with Windows-specific extras if on Windows)
- ✅ Installs `triton-windows==3.3.1.post21` on Windows
- ✅ Detects CUDA version using `nvidia-smi`
- ✅ Installs PyTorch 2.7.1 with appropriate CUDA support
- ✅ Installs torchvision, xformers, and other dependencies
- ✅ Installs additional packages (synthetic-data-kit, openai, sentence-transformers)

**Installation Time:** Approximately 5-15 minutes depending on your internet connection.

**Expected Output:**
```
============================================================
Package Installation with GPU Detection
============================================================
Operating System detected: Windows
Upgrading pip...
Windows detected. Installing unsloth[windows]...
Installing triton-windows==3.3.1.post21 for Windows...
Detecting CUDA version...
CUDA version detected: 11.8
Installing PyTorch 2.7.1 with CUDA 11.8 support
...
Installation completed successfully
============================================================
```

### Step 4: Verify Installation

```bash
# Check if PyTorch is installed correctly
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# If GPU is available, you should see:
# PyTorch: 2.7.1+cu118 (or similar)
# CUDA Available: True
```

### Optional: Use a `.env` File for Secrets

You can place secrets in a `.env` file at the repository root (auto-loaded by `run.py` if present). Values in the shell environment take precedence.

Example `.env`:
```
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token
```

## 🚀 Quick Start

### Basic Usage

After installation, you can run experiments:

```bash
# Make sure virtual environment is activated
# Windows: .\venv\Scripts\activate
# Linux: source venv/bin/activate

# Run a single experiment configuration
python run.py configs/config_gemma3.json
```

### With Custom Parameters

```bash
python run.py configs/config_gemma3.json \
  --threshold-f1 0.3 \
  --threshold-latency 0.5 \
  --factors accuracy latency
```

## 📁 Project Structure

```
finetuning-system/
├── src/
│   ├── data/                      # Data preparation modules
│   │   ├── __init__.py
│   │   └── data_preparation.py    # Dataset loading and QA generation
│   ├── models/                    # Model fine-tuning
│   │   ├── __init__.py
│   │   └── finetuner.py          # Training and evaluation logic
│   ├── engine/                    # Experiment execution
│   │   ├── __init__.py
│   │   ├── rule_engine.py        # Rule-based experiment orchestration
│   │   └── recommendation_engine.py  # Model selection logic
│   ├── __init__.py
│   └── utils.py                   # Utility functions (package installation, GPU detection)
├── configs/                       # Configuration files
│   ├── config_gemma3.json        # Gemma model configuration
│   ├── config_qwen3.json         # Qwen model configuration
│   └── config_example_with_comments.jsonc
├── run.py                         # Main entry point
├── requirements.txt               # Python dependencies (reference only)
├── finetuning_system.log         # Auto-generated log file
├── output/                        # Training checkpoints and outputs
└── README.md                      # This file
```

## ⚙️ Configuration

### Configuration File Structure

Create a JSON configuration file in the `configs/` directory:

```json
{
  "dataset": {
    "type": "huggingface",
    "name": "databricks/databricks-dolly-15k",
    "path": "databricks/databricks-dolly-15k",
    "hf_token": "your_huggingface_token",
    "splitter": "csv",
    "input_fields": ["instruction", "context"],
    "output_fields": ["response"],
    "batch_config": {
      "first_batch": 0.04,
      "second_batch": 0.06,
      "third_batch": 0.08,
      "test_batch": 0.01
    }
  },
  "output_dir": "results",
  "system_prompt": "You are a helpful assistant.",
  "experiments": {
    "exp1": {
      "run_always": true,
      "train_batch": "first_batch",
      "model": {
        "model_name": "unsloth/gemma-3-270m-it",
        "chat_template": "gemma3",
        "max_seq_len": 2048,
        "rank": 64,
        "alpha": 128,
        "dropout": 0,
        "eval_batch_size": 4
      },
      "sft": {
        "batch_size": 8,
        "epochs": 2,
        "learning_rate": 2e-5,
        "logging_steps": 50,
        "eval_steps": 50,
        "save_steps": 50,
        "eval_accumulation_steps": 30,
        "early_stopping_criteria": false
      },
      "rules": []
    }
  }
}
```

**Secrets note:** You can omit `hf_token` and set `HF_TOKEN` or `HUGGINGFACE_TOKEN` in the environment instead.

**Baseline note:** By default, the system now runs an automatic `exp0` baseline evaluation (no fine-tuning) before configured experiments. Set `"auto_baseline": false` at the root of your config to disable it.

**Evaluation throughput note:** Set `model.eval_batch_size` (default `4`) to run evaluation generation in batches for faster baseline and experiment scoring on capable GPUs.

**Terminal decision note:** Final output now prints a decision summary in one of these categories: `model is finetunable`, `model validation not increasing`, or `model is scalable via RAG`.

**Experiment trace note:** During rule-based execution, terminal logs now print why each experiment continues or stops, and training diagnostics such as `validation loss is increasing`, `training loss is increasing`, and possible overfitting alerts.

### Dataset Configuration Options

#### 1. CSV/JSON File
```json
"dataset": {
  "type": "file",
  "path": "data.csv",
  "splitter": "csv",
  "input_fields": ["question"],
  "output_fields": ["answer"]
}
```

#### 2. HuggingFace Dataset
```json
"dataset": {
  "type": "huggingface",
  "name": "databricks/databricks-dolly-15k",
  "path": "databricks/databricks-dolly-15k",
  "hf_token": "your_token_here",
  "input_fields": ["instruction", "context"],
  "output_fields": ["response"]
}
```

#### 3. PDF Documents
```json
"dataset": {
  "type": "file",
  "splitter": "pdf",
  "path": "document.pdf",
  "pdf_config": {
    "llm_config": {
      "api_base": "https://api.openai.com/v1",
      "api_key": "your_api_key",
      "model_name": "gpt-4o-mini"
    },
    "chunk_size": 2048,
    "overlap": 200,
    "qa_pairs_per_chunk": 3,
    "max_generation_tokens": 512
  }
}
```

**Secrets note:** You can omit `api_key` and set `OPENAI_API_KEY` or `LLM_API_KEY` in the environment instead.

### Rule-based Experiment Execution

Define conditional experiments based on previous results:

```json
"exp2": {
  "run_always": false,
  "train_batch": "second_batch",
  "rules": [
    {
      "conditions": [
        { "left": "exp1.f1", "op": ">", "right": "exp2.f1" },
        { "left": "exp1.last_eval_loss", "op": "<", "right": "exp1.min_eval_loss" }
      ]
    }
  ]
}
```

**Available Metrics for Rules:**
- `f1`, `precision`, `recall`, `latency`
- `last_eval_loss`, `min_eval_loss`
- `last_train_loss`, `min_train_loss`

**Available Operators:**
- `>`, `<`, `>=`, `<=`, `==`, `!=`

## 💻 Usage

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `config_files` | str (required) | - | Path(s) to configuration JSON file(s) |
| `--install-packages` | flag | - | Install all required packages with GPU detection |
| `--threshold-f1` | float | 0.2 | F1 score threshold for model selection |
| `--threshold-latency` | float | 0.3 | Latency threshold for model selection |
| `--factors` | list | `['accuracy', 'latency']` | Optimization factors |

### Examples

```bash
# Install packages (first time setup)
python run.py --install-packages

# Basic run
python run.py configs/config_gemma3.json

# Multiple configs with custom thresholds
python run.py configs/config_gemma3.json configs/config_qwen3.json \
  --threshold-f1 0.35 \
  --threshold-latency 0.4

# Optimize for accuracy only
python run.py configs/config_gemma3.json --factors accuracy

# Optimize for latency only
python run.py configs/config_gemma3.json --factors latency
```

## 📊 Output Structure

Results are organized by dataset with date tracking:

```
results/
├── metrics_dataset_name.csv       # Metrics with date column
├── logs_dataset_name_exp1.csv     # Training logs for exp1
├── logs_dataset_name_exp2.csv     # Training logs for exp2
└── finetuning_system.log          # System logs

output/
└── checkpoint-{step}/            # Training checkpoints
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── tokenizer files
```

### Metrics CSV Format

| Column | Description |
|--------|-------------|
| `precision` | Token-level precision score |
| `recall` | Token-level recall score |
| `f1` | F1 score (harmonic mean of precision and recall) |
| `latency` | Average inference time in seconds |
| `exp` | Experiment name |
| `model` | Model name |
| `dataset` | Dataset name |
| `date` | Experiment date (YYYY-MM-DD) |

### HTML Report

After a run completes, the system generates an HTML report in the output directory:

```
results/
└── report_<dataset>.html
```

The report includes:
- Dataset and model metadata
- Per-experiment metrics (precision, recall, F1, latency)
- Experiment configurations
- Training/evaluation loss curves

## 🌐 Platform-Specific Notes

### Windows

**Features:**
- ✅ Automatic installation of `triton-windows==3.3.1.post21`
- ✅ Automatic multiprocessing fixes for Windows compatibility
- ✅ Works with any NVIDIA GPU (RTX, GTX, Quadro, etc.)
- ✅ Automatic memory optimization for low-memory GPUs

**Setup:**
```powershell
# Create and activate venv
python -m venv venv
.\venv\Scripts\activate

# Install packages
python run.py --install-packages

# Run experiments
python run.py configs/config_gemma3.json
```

**Note:** The system automatically:
- Disables optimized loss functions that require Triton compilation
- Forces single-process mode for dataset processing
- Adjusts batch size for GPUs with < 6GB memory

### Linux

**Setup:**
```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install packages
python run.py --install-packages

# Run experiments
python run.py configs/config_gemma3.json
```

### Google Colab

```python
# Clone repository
!git clone <your-repo-url>
%cd finetuning-system

# Install packages
!python run.py --install-packages

# Run experiments
!python run.py configs/config_gemma3.json
```

### Kaggle Notebooks

1. **Enable GPU**: Settings → Accelerator → GPU T4 x2
2. **Enable Internet**: Settings → Internet → ON
3. **Upload files** to `/kaggle/working/`
4. **Run**:

```python
!python /kaggle/working/finetuning-system/run.py --install-packages
!python /kaggle/working/finetuning-system/run.py configs/config_gemma3.json
```

## 🔬 Advanced Features

### GPU Support

The system automatically:
- ✅ Detects CUDA-enabled GPUs (any NVIDIA GPU)
- ✅ Determines compatible CUDA version from driver
- ✅ Installs PyTorch with appropriate CUDA support
- ✅ Falls back to CPU if no GPU available
- ✅ Logs GPU memory and CUDA version
- ✅ Uses mixed precision training (FP16/BF16)
- ✅ Automatically adjusts batch size for low-memory GPUs (< 6GB)

### Memory Management

For GPUs with less than 6GB memory:
- Automatically reduces batch size
- Increases gradient accumulation to maintain effective batch size
- Enables memory-efficient allocation

### Logging System

All operations are logged to:
- **Console**: Real-time output
- **File**: `finetuning_system.log` (persistent)

Log format: `YYYY-MM-DD HH:MM:SS - LEVEL - MESSAGE`

### Evaluation Metrics

1. **Token F1 Score**: Measures token-level overlap between prediction and reference
2. **Semantic Accuracy**: Uses sentence embeddings for semantic similarity
3. **Latency**: Tracks inference time per sample
4. **Training Metrics**: Loss curves and evaluation metrics

### Early Stopping

Enable early stopping to prevent overfitting:

```json
"sft": {
  "early_stopping_criteria": true
}
```

Stops training if evaluation loss doesn't improve for 5 consecutive evaluations.

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution:**
- The system automatically reduces batch size for GPUs < 6GB
- Manually reduce batch size in config: `"batch_size": 2` or `1`
- Reduce `max_seq_len` in model config

#### 2. Package Installation Fails

**Solution:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Try installing packages again
python run.py --install-packages

# If specific package fails, install manually:
pip install <package-name>
```

#### 3. Import Errors

**Solution:**
```bash
# Make sure virtual environment is activated
# Windows: .\venv\Scripts\activate

# Reinstall packages
python run.py --install-packages
```

#### 4. GPU Not Detected

**Solution:**
```bash
# Verify NVIDIA driver is installed
nvidia-smi

# If nvidia-smi fails, install NVIDIA drivers:
# Windows: https://www.nvidia.com/Download/index.aspx
# Linux: Use your distribution's package manager
```

#### 5. Windows-Specific Issues

**Triton Compilation Errors:**
- Automatically handled by the system
- The system disables optimized loss functions on Windows
- Uses standard PyTorch cross-entropy loss instead

**Multiprocessing Errors:**
- Automatically handled by the system
- Forces single-process mode on Windows

#### 6. Slow Training

**Solutions:**
- Ensure GPU is being used (check logs for "GPU detected")
- Reduce `max_seq_len` in model config
- Increase `batch_size` if memory allows
- Check GPU utilization: `nvidia-smi` during training

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Update README.md with new features

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [HuggingFace](https://huggingface.co/) for transformers and datasets
- [TRL](https://github.com/huggingface/trl) for SFT training
