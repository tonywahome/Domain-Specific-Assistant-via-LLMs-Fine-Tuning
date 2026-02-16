# Domain-Specific Financial Assistant via LLM Fine-Tuning

A comprehensive project for fine-tuning **Google Gemma-2B** using **QLoRA** (4-bit quantization) on the Financial-QA-10k dataset to create a domain-specific assistant for financial question answering from SEC 10-K filings.

---

## ⚡ **Quick Start with Kaggle** (Recommended - FREE GPU!)

**Get started in 5 minutes:**

1. 📖 **Read**: [KAGGLE_QUICK_START.md](KAGGLE_QUICK_START.md) - Simple 5-step guide
2. 📋 **Detailed Setup**: [KAGGLE_SETUP.md](KAGGLE_SETUP.md) - Comprehensive instructions with troubleshooting
3. 🚀 **Run**: Upload `notebooks/financial_assistant_training.ipynb` to Kaggle with GPU enabled
4. ⏱️ **Wait**: ~2 hours of training
5. 💾 **Download**: Your fine-tuned model!

**All you need**: Kaggle account + HuggingFace token (both free)

---

## 🎯 Project Overview

This project demonstrates state-of-the-art parameter-efficient fine-tuning techniques to adapt a general-purpose Large Language Model for specialized financial domain tasks. The fine-tuned model can extract and synthesize information from complex financial documents to answer domain-specific questions.

### Key Features

- **Model**: Google Gemma-2B (2 billion parameters)
- **Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Dataset**: Financial-QA-10k (5,000 curated Q&A pairs from SEC 10-K filings)
- **Format**: Alpaca instruction-response template
- **Hardware**: Optimized for consumer GPUs (8-12GB VRAM) and free cloud platforms (Kaggle, Colab)
- **Companies**: NVIDIA, Apple, Tesla, Procter & Gamble, Kroger, Las Vegas Sands

## 📊 Dataset

The **Financial-QA-10k** dataset contains 7,001 high-quality question-answer pairs extracted from 2023 SEC 10-K filings of 6 major corporations across diverse sectors:

| Ticker | Company          | Sector               | Examples |
| ------ | ---------------- | -------------------- | -------- |
| NVDA   | NVIDIA           | Technology/AI        | ~1,167   |
| AAPL   | Apple            | Consumer Electronics | ~1,167   |
| TSLA   | Tesla            | Electric Vehicles    | ~1,167   |
| PG     | Procter & Gamble | Consumer Goods       | ~1,167   |
| KR     | Kroger           | Retail/Grocery       | ~1,167   |
| LVS    | Las Vegas Sands  | Gaming/Hospitality   | ~1,167   |

**Data Structure**:

- `question`: Financial question about the company
- `answer`: Ground truth answer extracted from filing
- `context`: Relevant excerpt from 10-K filing (200-500+ characters)
- `ticker`: Company stock ticker symbol
- `filing`: Filing type and year (2023_10K)

---

## 🚀 Getting Started

### 🌟 Recommended: Kaggle (Free GPU!)

**Complete Kaggle guides available:**

- 📖 **[KAGGLE_QUICK_START.md](KAGGLE_QUICK_START.md)** - Simple 5-step guide (start here!)
- 📋 **[KAGGLE_SETUP.md](KAGGLE_SETUP.md)** - Detailed instructions with troubleshooting

**Quick Summary**:

1. Upload dataset to Kaggle, create notebook with GPU
2. Add HuggingFace token for Gemma access
3. Upload `notebooks/financial_assistant_training.ipynb`
4. Click "Run All" and wait ~2 hours
5. Download your trained financial assistant model!

✅ **Perfect for**: Free GPU access, no local setup required  
⏱️ **Time**: ~2 hours of automated training

### Alternative Options

**Google Colab**: Upload notebook, enable GPU, run cells (limited free GPU time)

**Local Training**: Requires Python 3.10+, CUDA GPU (12GB+ VRAM), 20GB disk

```bash
git clone https://github.com/tonywahome/Domain-Specific-Assistant-via-LLMs-Fine-Tuning.git
cd Domain-Specific-Assistant-via-LLMs-Fine-Tuning
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/financial_assistant_training.ipynb
```

---

## 📁 Project Structure

```
Domain-Specific-Assistant-via-LLMs-Fine-Tuning/
├── dataset/
│   └── Financial-QA-10k.csv          # Raw dataset (7,001 examples)
├── data/
│   └── processed/                     # Preprocessed JSONL files
│       ├── train.jsonl                #   - Training set (4,500)
│       ├── validation.jsonl           #   - Validation set (250)
│       ├── test.jsonl                 #   - Test set (250)
│       └── metadata.json              #   - Dataset statistics
├── scripts/
│   └── data_preprocessing.py          # Standalone preprocessing script
├── notebooks/
│   └── financial_assistant_training.ipynb  # Complete training workflow
├── models/
│   ├── checkpoints/                   # Training checkpoints
│   └── final/                         # Final trained LoRA adapters
│       └── gemma-2b-financial-qa-lora/
├── outputs/
│   ├── logs/                          # Training logs
│   └── results/                       # Evaluation results
│       └── evaluation_results.json
├── configs/                           # (Integrated into notebook)
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── LICENSE                            # MIT License
```

## 🔧 Technical Details

### Model Architecture

- **Base Model**: Google Gemma-2B
  - 2 billion parameters
  - Transformer decoder architecture
  - 256K vocabulary size
  - 8192 token context window

### QLoRA Configuration

**4-bit Quantization** (BitsAndBytes):

- Quantization type: NF4 (4-bit NormalFloat)
- Compute dtype: bfloat16
- Double quantization: Enabled
- Memory reduction: ~75% compared to FP32

**LoRA Parameters**:

- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention layers)
- Trainable parameters: **~1.2%** of total model parameters

### Training Configuration

```python
Training Arguments:
  - Epochs: 3
  - Batch size per device: 4
  - Gradient accumulation steps: 4
  - Effective batch size: 16
  - Learning rate: 2e-4
  - LR scheduler: Cosine with warmup
  - Warmup steps: 100
  - Optimizer: Paged AdamW 8-bit
  - Mixed precision: BF16
  - Gradient checkpointing: Enabled
  - Max sequence length: 2048 tokens
```

### Data Preprocessing Pipeline

1. **Sampling**: 5,000 examples stratified by company ticker
2. **Normalization**:
   - Strip whitespace and control characters
   - Standardize financial notation ($X.XX billion)
   - Remove duplicate spaces
3. **Formatting**: Convert to Alpaca template
   - Instruction: Question
   - Input: Context from 10-K filing
   - Output: Answer
4. **Tokenization**: Verify sequences fit within 2048 token limit
5. **Truncation**: Smart context truncation at sentence boundaries
6. **Splitting**: 90% train, 5% validation, 5% test

### Alpaca Format Template

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{question}

### Input:
{context from 10-K filing}

### Response:
{answer}
```

## 📈 Expected Performance

### Training Metrics

- **Training time**: ~1-2 hours on T4 GPU, ~30-45 minutes on A100
- **Memory usage**: ~8-10GB VRAM (QLoRA), ~16GB VRAM (standard LoRA)
- **Training loss**: Decreases from ~2.5 to ~0.8-1.2
- **Validation loss**: Stabilizes around ~1.0-1.4

### Evaluation Metrics

Target performance on test set:

- **ROUGE-L**: > 0.40 (measures answer overlap with ground truth)
- **ROUGE-1**: > 0.45 (unigram overlap)
- **ROUGE-2**: > 0.25 (bigram overlap)

### Sample Output

**Question**: "What was NVIDIA's total revenue in fiscal 2023?"

**Context**: "Revenue for fiscal 2023 was $26.97 billion, up 61% from fiscal 2022. This growth was primarily driven by data center revenue..."

**Model Response**: "NVIDIA's total revenue in fiscal 2023 was $26.97 billion, representing a 61% increase compared to the previous fiscal year."

## 🎓 Use Cases

1. **Financial Document Analysis**: Automate extraction of key metrics from 10-K filings
2. **Investor Q&A Systems**: Provide instant answers to investors about company financials
3. **Compliance Monitoring**: Track financial disclosures and regulatory requirements
4. **Research Assistance**: Help analysts quickly find information in lengthy documents
5. **Educational Tools**: Teach financial analysis through interactive question-answering

## 🔄 Future Enhancements

- [ ] Fine-tune on full 7,000+ dataset
- [ ] Experiment with Gemma-7B for improved performance
- [ ] Add Retrieval-Augmented Generation (RAG) for real-time 10-K analysis
- [ ] Implement multi-year temporal analysis (compare 2022 vs 2023 filings)
- [ ] Create REST API with FastAPI for production deployment
- [ ] Build Gradio/Streamlit web interface
- [ ] Expand to include 10-Q quarterly reports
- [ ] Add numerical reasoning validation layer
- [ ] Fine-tune on multiple financial document types (earnings calls, prospectuses)

## 📝 Model Usage

After training, use the model in your own code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model with 4-bit quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "models/final/gemma-2b-financial-qa-lora"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "models/final/gemma-2b-financial-qa-lora"
)

# Inference function
def ask_financial_question(question, context):
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{question}

### Input:
{context}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

# Example usage
question = "What was the company's operating margin?"
context = "Operating margin for fiscal 2023 improved to 32%, up from 28% in the prior year..."

answer = ask_financial_question(question, context)
print(answer)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Dataset expansion (more companies, years, document types)
- Hyperparameter optimization experiments
- Additional evaluation metrics
- Production deployment examples
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google** for the Gemma model
- **Hugging Face** for transformers, peft, and trl libraries
- **SEC EDGAR** for public financial filings
- **Financial-QA-10k Dataset** creators
- **Kaggle** for free GPU compute resources

## 📧 Contact

- **Author**: Tony Wahome
- **Repository**: [Domain-Specific-Assistant-via-LLMs-Fine-Tuning](https://github.com/tonywahome/Domain-Specific-Assistant-via-LLMs-Fine-Tuning)

---

**Note**: This project is for educational and research purposes. Financial information extracted by the model should be verified against official SEC filings before making any investment decisions.

## 🔗 Additional Resources

- [Gemma Model Card](https://ai.google.dev/gemma)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Alpaca Format](https://github.com/tatsu-lab/stanford_alpaca)
- [SEC EDGAR Database](https://www.sec.gov/edgar)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)

---

**⭐ If you find this project helpful, please consider giving it a star!**
