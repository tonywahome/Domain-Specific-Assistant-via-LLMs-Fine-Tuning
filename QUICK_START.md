# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

This guide will help you quickly set up and run the Financial Assistant fine-tuning project.

## Prerequisites

- **Hardware**: Kaggle
- **Software**: Python 3.10+, CUDA toolkit (for local training)

## Step 1: Choose Your Platform

### Option A: Kaggle (Easiest - Recommended)

1. **Sign up** at [kaggle.com](https://www.kaggle.com)
2. **Create a new notebook**
3. **Enable GPU**: Settings → Accelerator → GPU T4 x2
4. **Upload dataset**:
   - Go to "Add Data" → "Upload"
   - Upload `dataset/Financial-QA-10k.csv`
5. **Copy notebook**:
   - Upload `notebooks/financial_assistant_training.ipynb`
   - Or copy-paste cell contents
6. **Run All**: Click "Run All" and wait ~1-2 hours
7. **Download model**: From `models/final/` folder

✅ **Pros**: Free GPU, no setup, pre-installed libraries
❌ **Cons**: Session time limits, internet required

```bash
# 1. Clone repository
git clone https://github.com/tonywahome/Domain-Specific-Assistant-via-LLMs-Fine-Tuning.git
cd Domain-Specific-Assistant-via-LLMs-Fine-Tuning

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Jupyter
jupyter notebook notebooks/financial_assistant_training.ipynb
```

## Step 2: Run the Training Notebook

The notebook is organized into sections that run sequentially:

### Section 1: Environment Setup (2-3 minutes)

- Installs required packages
- Imports libraries
- Checks GPU availability

### Section 2: Data Preprocessing (3-5 minutes)

- Loads Financial-QA-10k dataset
- Samples 5,000 examples
- Converts to Alpaca format
- Splits into train/val/test

### Section 3: Model Configuration (5-10 minutes)

- Loads Gemma-2B with 4-bit quantization
- Configures QLoRA parameters
- Prepares model for training

### Section 4: Training (60-120 minutes)

- Fine-tunes the model
- Monitors loss curves
- Saves checkpoints

### Section 5: Inference & Evaluation (5-10 minutes)

- Tests model on sample questions
- Calculates ROUGE metrics
- Saves results

## Step 3: Use Your Model

After training completes, you can:

### Test Interactively

Run the last cells of the notebook to test custom questions:

```python
question = "What was the company's revenue?"
context = "The company reported revenue of $26.97 billion..."

response = generate_response(question, context)
print(response)
```

### Download Model

The trained LoRA adapters are saved to:

```
models/final/gemma-2b-financial-qa-lora/
```

### Load in Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "path/to/gemma-2b-financial-qa-lora"
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
```

## Troubleshooting

### ❌ Out of Memory Error

**Solutions**:

1. Reduce `per_device_train_batch_size` from 4 to 2 or 1
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `MAX_SEQ_LENGTH` from 2048 to 1024

### ❌ Model Download Fails

**Solution**: You may need HuggingFace authentication:

```python
from huggingface_hub import login
login(token="YOUR_TOKEN_HERE")
```

Get token from: https://huggingface.co/settings/tokens

### ❌ CUDA Out of Memory

**Solution**: Restart runtime and reduce batch size:

```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Changed from 4
    gradient_accumulation_steps=16,  # Changed from 4
    # ... other args
)
```

### ❌ Dataset Not Found

**Solution**: Update the path in the notebook:

```python
RAW_DATA_PATH = "/kaggle/input/your-dataset-name/Financial-QA-10k.csv"
```

## Expected Timeline

| Task                | Time           | Notes                   |
| ------------------- | -------------- | ----------------------- |
| Environment Setup   | 2-3 min        | First-time installation |
| Data Preprocessing  | 3-5 min        | Tokenization takes time |
| Model Loading       | 5-10 min       | Downloading weights     |
| Training (3 epochs) | 60-120 min     | Depends on GPU          |
| Evaluation          | 5-10 min       | Running on test set     |
| **Total**           | **75-150 min** | ~1.5-2.5 hours          |

## Hardware Requirements

### Minimum (QLoRA with this config):

- **GPU**: 8GB VRAM (GTX 1080, RTX 2070, T4)
- **RAM**: 16GB system RAM
- **Disk**: 20GB free space

### Recommended:

- **GPU**: 12-16GB VRAM (RTX 3090, 4090, A4000, A5000)
- **RAM**: 32GB system RAM
- **Disk**: 50GB SSD storage

### Cloud Alternatives:

- **Kaggle**: Free T4 GPU (16GB VRAM), 30h/week
- **Colab**: Free T4 (limited hours)
- **Colab Pro**: $10/month, better GPUs
- **Lambda Labs**: $0.50/hour for A10 (24GB)

## Next Steps

After successful training:

1. **Experiment**: Try different hyperparameters
2. **Expand**: Use full 7,000 examples instead of 5,000
3. **Upgrade**: Try Gemma-7B for better performance
4. **Deploy**: Create API or web interface
5. **Integrate**: Add RAG for document retrieval

## Getting Help

- **Issues**: File on [GitHub Issues](https://github.com/tonywahome/Domain-Specific-Assistant-via-LLMs-Fine-Tuning/issues)
- **Documentation**: See [README.md](README.md)
- **Community**: HuggingFace forums, Reddit r/MachineLearning

---

**Ready? Open the notebook and click "Run All"! 🚀**
