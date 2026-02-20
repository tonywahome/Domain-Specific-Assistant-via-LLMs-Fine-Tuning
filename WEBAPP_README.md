# Financial Assistant AI - Web App

## 🚀 Quick Start

Run the web interface locally:

### Windows

```bash
run_app.bat
```

### Linux/Mac

```bash
chmod +x run_app.sh
./run_app.sh
```

### Manual Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at: http://localhost:8501

## 📱 Features

- **Intuitive Finance-themed Interface**: Professional green color scheme
- **Real-time Response Generation**: Ask financial questions and get instant answers
- **Context Support**: Paste excerpts from financial reports for better responses
- **Adjustable Parameters**: Control response length, creativity, and diversity
- **Question History**: Track your previous queries
- **Example Questions**: Quick access to common financial queries
- **Model Information**: View details about the fine-tuned model

## 🎨 Screenshot Preview

The interface includes:

- 💰 Financial-themed header with gradient design
- 📊 Question input area with optional context
- ⚙️ Sidebar with generation parameters
- 💡 Pre-loaded example questions
- 📈 Quick statistics dashboard
- 📜 Question history viewer

## 🔧 Customization

### Change Model

Edit `app.py` line 58:

```python
MODEL_ID = "your-username/your-model-name"
```

### Adjust Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#2e7d32"  # Change colors here
```

### Modify Parameters

Default generation settings in the sidebar:

- Max Response Length: 256 tokens
- Temperature: 0.7
- Top-p: 0.9

## 📦 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions including:

- Streamlit Community Cloud (Free)
- HuggingFace Spaces
- AWS/GCP/Azure
- Docker deployment

## 🐛 Troubleshooting

### App won't start

```bash
pip install --upgrade streamlit transformers
```

### Out of memory

Lower the `max_new_tokens` slider to reduce memory usage

### Slow responses

- Use GPU if available
- Reduce response length
- Use lower temperature for faster generation

## 📚 Model Information

- **Base Model**: Google Gemma-2B
- **Fine-tuning**: QLoRA (4-bit quantization)
- **Dataset**: Financial-QA-10K (3,000+ examples)
- **HuggingFace**: [Antonomics/gemma-2b-financial-qa-lora](https://huggingface.co/Antonomics/gemma-2b-financial-qa-lora)

## 📞 Support

For issues or questions:

- Create an issue on GitHub
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guides
- Review the main [README.md](README.md)

---

**Built with ❤️ using Streamlit and HuggingFace**
