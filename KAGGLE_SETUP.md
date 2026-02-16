# 🚀 Kaggle Deployment Guide

## Step-by-Step Instructions for Running on Kaggle

This guide walks you through deploying the Financial Assistant fine-tuning project on Kaggle's free GPU platform.

---

## ✅ Prerequisites

1. **Kaggle Account**: Sign up at [kaggle.com](https://www.kaggle.com) (free)
2. **Phone Verification**: Required for GPU access
3. **HuggingFace Account**: Sign up at [huggingface.co](https://huggingface.co) (free)
4. **Files Ready**: `dataset/Financial-QA-10k.csv` and `notebooks/financial_assistant_training.ipynb`

---

## 📋 Step 1: Create HuggingFace Access Token

Before starting, you'll need a token to download the Gemma model:

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Name it: `kaggle-gemma-access`
4. Type: Select **"Read"**
5. Click **"Generate a token"**
6. **Copy the token** (you'll need it later)
7. **Accept Gemma license**: Visit [google/gemma-2b](https://huggingface.co/google/gemma-2b) and accept terms

---

## 📂 Step 2: Upload Dataset to Kaggle

### Option A: Upload as Dataset (Recommended)

1. **Go to Kaggle Datasets**: [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **"New Dataset"**
3. **Upload file**:
   - Click "Upload"
   - Select `Financial-QA-10k.csv` from your `dataset/` folder
   - Wait for upload to complete (~10MB, takes 1-2 minutes)
4. **Configure dataset**:
   - Title: `Financial QA 10k Dataset`
   - Subtitle: `SEC 10-K Filing Q&A Pairs for Financial LLM Fine-tuning`
   - Description: `7,001 question-answer pairs from 2023 SEC 10-K filings of NVDA, AAPL, TSLA, PG, KR, LVS`
   - Visibility: **Public** or **Private** (your choice)
5. Click **"Create"**
6. **Copy the dataset path** (e.g., `yourusername/financial-qa-10k-dataset`)

### Option B: Upload Directly to Notebook

You can also upload files directly when creating the notebook (see Step 3).

---

## 📓 Step 3: Create Kaggle Notebook

1. **Go to Kaggle Notebooks**: [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. **Configure notebook**:
   - Click the **"File"** menu
   - Select **"Import notebook"**
   - Upload `notebooks/financial_assistant_training.ipynb`

   OR create new notebook and copy-paste all cells manually

4. **Set notebook title**: `Financial Assistant Fine-tuning with Gemma-2B`

---

## ⚙️ Step 4: Configure Notebook Settings

### Enable GPU Accelerator

1. Click **"Accelerator"** dropdown (top right, near "Run")
2. Select: **"GPU T4 x2"** (recommended) or **"GPU P100"**
3. GPU status should show: ✅ **GPU Enabled**

### Add Dataset

1. Click **"Add data"** button (right sidebar)
2. Search for your dataset: `financial-qa-10k-dataset` or your username
3. Click **"Add"** next to your dataset
4. The dataset path will be: `/kaggle/input/financial-qa-10k-dataset/Financial-QA-10k.csv`

### Enable Internet Access

1. Click **"Settings"** (right sidebar)
2. Under **"Internet"**, toggle: **ON** ✅
3. This is required to download Gemma model from HuggingFace

### Set Persistence

1. In **"Settings"**, enable **"Notebook persistence"**
2. This saves your progress if session times out

---

## 🔑 Step 5: Add HuggingFace Token

In the **first code cell** of the notebook, add your HuggingFace token:

```python
# HuggingFace Authentication
from huggingface_hub import login

# Replace with your actual token from Step 1
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxx"  # ⚠️ PASTE YOUR TOKEN HERE
login(token=HF_TOKEN)

print("✓ Authenticated with HuggingFace!")
```

⚠️ **Security Note**: For private notebooks only. Don't share notebooks with tokens publicly!

---

## 🔧 Step 6: Update Dataset Path

Find the cell with `RAW_DATA_PATH` configuration and update it:

```python
# Configuration
RAW_DATA_PATH = "/kaggle/input/financial-qa-10k-dataset/Financial-QA-10k.csv"  # ⚠️ UPDATE THIS
MODEL_NAME = "google/gemma-2b"
MAX_SAMPLES = 5000
MAX_SEQ_LENGTH = 2048
```

Replace `financial-qa-10k-dataset` with your actual dataset name.

---

## ▶️ Step 7: Run the Notebook

### Option A: Run All Cells (Easiest)

1. Click **"Run All"** button at the top
2. Monitor progress in the output cells
3. Training will take **1-2 hours** on T4 GPU
4. Don't close the browser tab during training

### Option B: Run Cell by Cell

1. Click **"Run"** button on each cell sequentially
2. Wait for each cell to complete before running next
3. Useful for debugging or understanding each step

### What to Expect

| Section            | Time            | What Happens                |
| ------------------ | --------------- | --------------------------- |
| Environment Setup  | 2-3 min         | Installs packages           |
| Data Preprocessing | 3-5 min         | Loads & formats dataset     |
| Model Loading      | 5-10 min        | Downloads Gemma-2B (2GB)    |
| Training           | 60-120 min      | Fine-tunes model (3 epochs) |
| Evaluation         | 5-10 min        | Tests on test set           |
| **Total**          | **~90-150 min** | **1.5-2.5 hours**           |

---

## 📊 Step 8: Monitor Training

### Check GPU Usage

Run this in a new cell to monitor GPU:

```python
!nvidia-smi
```

You should see:

- GPU: Tesla T4 or P100
- Memory usage: ~10-12GB / 16GB
- Processes: Python using GPU

### Watch Training Metrics

During training, you'll see:

```
[10/xxx] loss: 2.456
[20/xxx] loss: 2.123
[30/xxx] loss: 1.892
...
[500/xxx] eval_loss: 1.234 (evaluation checkpoint)
```

✅ **Good training**: Loss decreases steadily from ~2.5 to ~1.0
❌ **Problem**: Loss increases or stays flat (check learning rate)

### Training Progress Indicators

```
Epoch 1/3
[=======>..................] 30% | Loss: 1.85 | ETA: 45min

Epoch 2/3
[=============>............] 60% | Loss: 1.42 | ETA: 25min

Epoch 3/3
[=====================>....] 90% | Loss: 1.18 | ETA: 5min
```

---

## 💾 Step 9: Save Your Model

After training completes, the model is automatically saved to:

```
/kaggle/working/models/final/gemma-2b-financial-qa-lora/
```

### Download Model Files

1. Click the **"Output"** tab (right sidebar)
2. Navigate to `models/final/gemma-2b-financial-qa-lora/`
3. You'll see files:
   - `adapter_config.json`
   - `adapter_model.safetensors` or `adapter_model.bin`
   - `tokenizer_config.json`
   - `special_tokens_map.json`
   - etc.
4. Click **"⋮"** → **"Download"** for each file

OR download entire folder as ZIP:

```python
# Add this cell at the end
import shutil
shutil.make_archive(
    '/kaggle/working/gemma-2b-financial-lora',
    'zip',
    '/kaggle/working/models/final/gemma-2b-financial-qa-lora'
)
print("✓ Model archived: gemma-2b-financial-lora.zip")
```

Then download the ZIP from the Output tab.

---

## 🧪 Step 10: Test Your Model

Run the inference cells to test:

```python
# Test with sample from test set
example = test_data[0]

print(f"Question: {example['instruction']}")
print(f"Context: {example['input'][:200]}...")

response = generate_response(
    example['instruction'],
    example['input'],
    max_new_tokens=200
)

print(f"\nModel Response:\n{response}")
print(f"\nGround Truth:\n{example['output']}")
```

### Sample Questions to Try

```python
# Custom test 1: Revenue question
generate_response(
    "What was the company's total revenue?",
    "Revenue for fiscal 2023 was $26.97 billion, up 61% from fiscal 2022..."
)

# Custom test 2: Margin question
generate_response(
    "What was the operating margin?",
    "Operating margin improved to 32% in fiscal 2023, compared to 28% in the prior year..."
)
```

---

## 📈 Step 11: View Evaluation Results

After evaluation completes, check metrics:

```
EVALUATION RESULTS
========================================
ROUGE Scores (on 100 test examples):
  - ROUGE-1: 0.4532
  - ROUGE-2: 0.2678
  - ROUGE-L: 0.4123
  - ROUGE-Lsum: 0.4098
```

✅ **Good performance**: ROUGE-L > 0.40
⚠️ **Needs improvement**: ROUGE-L < 0.30

Results are saved to: `/kaggle/working/outputs/results/evaluation_results.json`

---

## 🚀 Step 12: Share or Deploy (Optional)

### Option A: Make Notebook Public

1. Click **"Share"** button
2. Toggle **"Public"**
3. Add description and tags
4. Others can now view your work!

### Option B: Push to HuggingFace Hub

Add this cell to upload your model:

```python
# Push to HuggingFace Hub
model_id = "your-username/gemma-2b-financial-qa"

trainer.model.push_to_hub(model_id, token=HF_TOKEN)
tokenizer.push_to_hub(model_id, token=HF_TOKEN)

print(f"✓ Model uploaded: https://huggingface.co/{model_id}")
```

### Option C: Create Model Card

Document your model on HuggingFace with:

- Training details
- Performance metrics
- Usage examples
- Limitations

---

## ⚠️ Troubleshooting

### Problem: "Out of Memory" Error

**Solution 1**: Reduce batch size

```python
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Changed from 4
    gradient_accumulation_steps=8,  # Changed from 4
    # ... other settings
)
```

**Solution 2**: Use smaller sequence length

```python
MAX_SEQ_LENGTH = 1024  # Changed from 2048
```

**Solution 3**: Restart kernel and clear cache

```python
import torch
torch.cuda.empty_cache()
```

### Problem: "ValueError: You can't train a model that has been loaded in 8-bit or 4-bit precision on a different device"

**Full Error**:

```
ValueError: You can't train a model that has been loaded in 8-bit or 4-bit precision on a different device than the one you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device()}`
```

**Cause**: When using 4-bit quantization (QLoRA), the model must be loaded on a specific GPU device, not with `device_map="auto"`

**Solution**: The notebook has been updated with the fix. Ensure your model loading cell uses:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},  # Force to GPU 0 (CORRECTED)
    trust_remote_code=True,
)
```

**NOT** ❌:

```python
device_map="auto",  # This causes the error
```

If you're using an older version of the notebook, update this line in the model loading cell.

### Problem: "Model download fails"

**Cause**: Missing or invalid HuggingFace token

**Solution**:

1. Verify token is correct
2. Check you accepted Gemma license at [google/gemma-2b](https://huggingface.co/google/gemma-2b)
3. Ensure Internet is enabled in notebook settings

### Problem: "Dataset not found"

**Solution**: Update path

```python
import os
print(os.listdir('/kaggle/input'))  # List all datasets
```

Then update `RAW_DATA_PATH` to match actual path.

### Problem: "Session timeout"

**Cause**: Kaggle sessions timeout after 9 hours idle or 12 hours active

**Solution**:

1. Enable "Notebook persistence" in settings
2. Checkpoints are saved every 500 steps to `/kaggle/working/models/checkpoints/`
3. Resume from last checkpoint if needed

### Problem: "Kernel crash"

**Solution**:

1. Restart kernel: **"Restart & Run All"**
2. If persists, reduce memory usage (smaller batch size)
3. Try P100 GPU instead of T4 (different architecture)

---

## 📊 Resource Limits

**Kaggle Free Tier**:

- ✅ GPU time: 30 hours/week
- ✅ Session: 9 hours idle / 12 hours active
- ✅ Disk: 20GB temporary storage
- ✅ RAM: 13GB (with GPU T4)
- ✅ Internet: Enabled (required)

**This project uses**:

- ⏱️ ~2 hours GPU time
- 💾 ~8-10GB disk space
- 🧠 ~10-12GB RAM
- All within free tier limits! ✅

---

## 🎯 Success Checklist

Before considering training complete:

- [ ] All cells executed without errors
- [ ] Training loss decreased from ~2.5 to ~1.0
- [ ] Validation loss stable around ~1.2-1.4
- [ ] ROUGE-L score > 0.40 on test set
- [ ] Sample predictions look reasonable
- [ ] Model files saved to `/kaggle/working/models/final/`
- [ ] Downloaded model ZIP file
- [ ] (Optional) Pushed to HuggingFace Hub

---

## 🎓 Next Steps After Training

1. **Test thoroughly**: Try various financial questions
2. **Compare outputs**: Check against ground truth
3. **Error analysis**: Find where model struggles
4. **Iterate**: Adjust hyperparameters and retrain
5. **Deploy**: Create API or web interface
6. **Expand**: Try Gemma-7B or full dataset

---

## 📞 Getting Help

**Having issues?**

1. **Check logs**: Review cell outputs for error messages
2. **Restart kernel**: Often fixes transient issues
3. **Kaggle Forums**: [kaggle.com/discussions](https://www.kaggle.com/discussions)
4. **GitHub Issues**: File issue on project repository
5. **HuggingFace Forums**: For model-specific questions

---

## 🌟 Tips for Best Results

1. **Use T4 x2 GPU**: Faster than P100 for this workload
2. **Monitor actively**: Don't leave unattended for hours
3. **Save frequently**: Download checkpoints periodically
4. **Test incrementally**: Run few cells at a time first
5. **Check metrics**: Ensure loss decreases each epoch
6. **Document changes**: Note what works in markdown cells

---

## ✅ You're Ready!

Follow these steps and you'll have a fine-tuned financial assistant in ~2 hours. Good luck! 🚀

**Questions? Issues? Success stories?** Share in notebook comments or GitHub!

---

### Quick Command Reference

```python
# Check GPU
!nvidia-smi

# List input datasets
!ls /kaggle/input

# Check disk space
!df -h /kaggle/working

# Monitor training logs
!tail -f logs/training.log  # If logging to file

# Archive model
!zip -r model.zip models/final/

# Test token count
len(tokenizer.encode("Your text here"))
```

---

**Last updated**: February 16, 2026
