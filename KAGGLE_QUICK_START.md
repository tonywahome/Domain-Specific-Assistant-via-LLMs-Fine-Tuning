# 🎯 Kaggle Quick Deploy - 5 Steps

## You've chosen Kaggle! Here's what to do:

### ⚡ Quick Overview

1. **Upload dataset to Kaggle** (2 mins)
2. **Create notebook with GPU** (1 min)
3. **Add HuggingFace token** (1 min)
4. **Run all cells** (90-120 mins)
5. **Download trained model** (2 mins)

**Total time: ~2 hours** (mostly training time)

---

## 📋 Step-by-Step Instructions

### STEP 1: Get HuggingFace Token (First Time Only)

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"** → Name it "kaggle" → Type: **Read**
3. Copy the token (starts with `hf_...`)
4. Visit https://huggingface.co/google/gemma-2b and click **"Agree"** to license

### STEP 2: Upload Dataset to Kaggle

1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Drag and drop: `dataset/Financial-QA-10k.csv`
4. Title: `Financial QA 10k`
5. Click **"Create"**
6. Remember the dataset name (e.g., `yourusername/financial-qa-10k`)

### STEP 3: Create Kaggle Notebook

1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. **Important settings:**
   - Click **"Accelerator"** → Select **"GPU T4 x2"**
   - Click **"Settings"** → Toggle **"Internet"** to **ON**
   - Click **"Add data"** → Search for your dataset → Click **"Add"**

### STEP 4: Set Up Notebook

Copy the content from `notebooks/financial_assistant_training.ipynb`

**OR** for easier setup, add this as the FIRST cell:

```python
# Copy from: kaggle_config.py
# This sets up everything

!wget https://raw.githubusercontent.com/tonywahome/Domain-Specific-Assistant-via-LLMs-Fine-Tuning/main/kaggle_config.py
%run kaggle_config.py
```

Then **replace** `hf_YOUR_TOKEN_HERE` with your actual token from Step 1.

### STEP 5: Run Training

1. Click **"Run All"** button at top
2. Wait ~90-120 minutes for training
3. Monitor progress - loss should decrease
4. After completion, download model from **Output** tab

---

## 🔍 What the Notebook Does

| Section   | Time   | Description                           |
| --------- | ------ | ------------------------------------- |
| **Setup** | 3 min  | Install packages, load libraries      |
| **Data**  | 5 min  | Load & preprocess 5,000 Q&A pairs     |
| **Model** | 10 min | Download Gemma-2B (2GB), apply QLoRA  |
| **Train** | 90 min | Fine-tune for 3 epochs                |
| **Eval**  | 5 min  | Test on 250 examples, compute metrics |

---

## 📊 What to Watch During Training

### Good Signs ✅

```
Epoch 1/3: loss = 2.45 → 1.89
Epoch 2/3: loss = 1.85 → 1.42
Epoch 3/3: loss = 1.38 → 1.12
```

Loss decreasing = model learning!

### Bad Signs ❌

```
loss = 2.45 → 2.43 → 2.44 (not decreasing)
Out of Memory errors
GPU shows 0% utilization
```

If you see these, check **KAGGLE_SETUP.md** troubleshooting section.

---

## 💾 Download Your Model

After training:

1. Click **"Output"** tab (right sidebar)
2. Navigate to: `models/final/gemma-2b-financial-qa-lora/`
3. Download all files (or run archive cell for ZIP)

Files you'll get:

- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - Trained weights (~30MB)
- `tokenizer_config.json` - Tokenizer settings
- etc.

---

## 🎯 Testing Your Model

At the end of the notebook, you can test:

```python
# Ask a financial question
question = "What was the company's revenue in fiscal 2023?"
context = "Revenue was $26.97 billion, up 61% from prior year..."

response = generate_response(question, context)
print(response)
# Output: "The company's revenue in fiscal 2023 was $26.97 billion..."
```

Try your own questions!

---

## ⚠️ Common Issues & Fixes

### "Out of Memory"

→ In training config cell, change:

```python
per_device_train_batch_size=2  # was 4
gradient_accumulation_steps=8  # was 4
```

### "Dataset not found"

→ Check path matches your Kaggle dataset name:

```python
!ls /kaggle/input  # See what's available
RAW_DATA_PATH = "/kaggle/input/YOUR-DATASET-NAME/Financial-QA-10k.csv"
```

### "Model download fails"

→ Verify HuggingFace token and accepted Gemma license

### "No GPU detected"

→ Click **Accelerator** dropdown → Select **GPU T4 x2**

### "ValueError: different device" (4-bit training error)

→ **Fixed in latest notebook!** Re-download if you get this error.  
The model must use `device_map={"": 0}` not `device_map="auto"` for 4-bit training.

---

## 📈 Expected Results

After 2 hours of training:

- **ROUGE-L**: ~0.40-0.45 (measures answer quality)
- **Training loss**: ~1.0-1.2 (lower = better)
- **Model size**: ~30MB (just the LoRA adapters!)

The model can now answer financial questions based on 10-K filing context!

---

## 🚀 Next Steps

After successful training:

1. **Test thoroughly**: Try different companies/questions
2. **Share**: Make notebook public on Kaggle
3. **Upload to HuggingFace**: Share your model (optional)
4. **Iterate**: Try different hyperparameters
5. **Expand**: Use full 7,000 examples or Gemma-7B

---

## 📚 Full Documentation

- **Detailed Guide**: See `KAGGLE_SETUP.md` for comprehensive instructions
- **Troubleshooting**: Complete solutions for all error scenarios
- **Project Info**: Read `README.md` for technical details

---

## ✅ Ready to Start?

1. ✅ HuggingFace token ready?
2. ✅ Dataset uploaded to Kaggle?
3. ✅ Notebook created with GPU enabled?
4. ✅ Internet turned ON in settings?

**Yes to all?** → Click **"Run All"** and grab coffee! ☕

Training takes ~2 hours. You'll have a fine-tuned financial assistant when done! 🎉

---

**Need help?** Check `KAGGLE_SETUP.md` or open GitHub issue.

**Questions during training?** Comment on your Kaggle notebook for help from community.

Good luck! 🚀
