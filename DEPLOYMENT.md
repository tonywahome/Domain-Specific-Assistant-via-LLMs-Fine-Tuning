# 🚀 Financial Assistant AI - Deployment Guide

This guide explains how to deploy the Financial Assistant AI web interface using Streamlit.

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for GPU inference)
- Internet connection for downloading the model from HuggingFace

## 🔧 Local Deployment

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

The app will automatically:

- Download the model from HuggingFace: `Antonomics/gemma-2b-financial-qa-lora`
- Load the model with 4-bit quantization
- Start the web interface (usually at `http://localhost:8501`)

### 3. Access the Interface

Open your browser and navigate to:

```
http://localhost:8501
```

## ☁️ Cloud Deployment Options

### Option 1: Streamlit Community Cloud (Free)

1. **Push your code to GitHub** (already done if you're reading this!)

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `Domain-Specific-Assistant-via-LLMs-Fine-Tuning`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Configuration:**
   - Streamlit Cloud will automatically use `requirements.txt`
   - The app will be publicly accessible at: `https://[your-username]-financial-assistant.streamlit.app`

**Note:** Streamlit Community Cloud has limited resources. For heavy usage, consider paid options below.

### Option 2: HuggingFace Spaces

1. **Create a new Space:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Select "Streamlit" as the SDK
   - Name it (e.g., "financial-assistant-ai")

2. **Upload files:**

   ```bash
   git clone https://huggingface.co/spaces/[your-username]/financial-assistant-ai
   cd financial-assistant-ai
   cp /path/to/app.py .
   cp /path/to/requirements.txt .
   git add .
   git commit -m "Add Financial Assistant app"
   git push
   ```

3. **Your app will be live at:**
   ```
   https://huggingface.co/spaces/[your-username]/financial-assistant-ai
   ```

### Option 3: AWS/GCP/Azure (Production)

For production deployments with high traffic:

#### AWS EC2 Deployment

1. **Launch an EC2 instance:**
   - Instance type: `g4dn.xlarge` or larger (for GPU)
   - AMI: Deep Learning AMI (Ubuntu)
   - Storage: 50GB+ EBS volume

2. **SSH into the instance and setup:**

   ```bash
   # Clone repository
   git clone https://github.com/tonywahome/Domain-Specific-Assistant-via-LLMs-Fine-Tuning.git
   cd Domain-Specific-Assistant-via-LLMs-Fine-Tuning

   # Install dependencies
   pip install -r requirements.txt

   # Run with nohup for persistent execution
   nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
   ```

3. **Configure security group:**
   - Allow inbound traffic on port 8501
   - Access at: `http://[EC2-PUBLIC-IP]:8501`

4. **Add HTTPS with Nginx (optional):**

   ```bash
   # Install Nginx
   sudo apt update
   sudo apt install nginx

   # Configure reverse proxy
   sudo nano /etc/nginx/sites-available/financial-assistant
   ```

   Add configuration:

   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
       }
   }
   ```

#### Docker Deployment

1. **Create `Dockerfile`:**

   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       git \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy app files
   COPY app.py .

   # Expose Streamlit port
   EXPOSE 8501

   # Health check
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

   # Run the app
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run:**
   ```bash
   docker build -t financial-assistant .
   docker run -p 8501:8501 --gpus all financial-assistant
   ```

## 🎛️ Configuration Options

### Environment Variables

You can set these in a `.streamlit/config.toml` file:

```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 100

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501

[theme]
primaryColor = "#2e7d32"
backgroundColor = "#f8f9fa"
secondaryBackgroundColor = "#ffffff"
textColor = "#262730"
font = "sans serif"
```

### Model Configuration

To use a different model, modify `app.py`:

```python
MODEL_ID = "your-username/your-model-name"
```

## 🔒 Security Considerations

For production deployments:

1. **Add authentication:**

   ```bash
   pip install streamlit-authenticator
   ```

2. **Rate limiting:**
   - Implement request throttling
   - Use load balancers

3. **HTTPS:**
   - Always use HTTPS in production
   - Use Let's Encrypt for free SSL certificates

4. **API Keys:**
   - Store HuggingFace tokens in environment variables
   - Use secrets management (AWS Secrets Manager, etc.)

## 📊 Monitoring

### Streamlit Cloud

- Built-in analytics available in the dashboard
- View logs in real-time

### Custom Monitoring

Add to `app.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## ⚡ Performance Optimization

### For GPU Inference

- The app uses 4-bit quantization by default
- Requires ~4GB VRAM
- Response time: 2-5 seconds

### For CPU Inference

- Set `device_map="cpu"` in the model loading
- Expect 10-30 second response times
- Not recommended for production

### Caching

The app uses `@st.cache_resource` to load the model once and reuse it across sessions.

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size or use CPU inference:

```python
device_map = "cpu"  # Instead of "auto"
```

### Issue: Model download fails

**Solution:** Pre-download the model:

```bash
huggingface-cli download Antonomics/gemma-2b-financial-qa-lora
```

### Issue: Slow inference

**Solution:**

- Reduce `max_new_tokens` in the sidebar
- Use GPU if available
- Consider model distillation for faster inference

## 📞 Support

- **GitHub Issues:** [Create an issue](https://github.com/tonywahome/Domain-Specific-Assistant-via-LLMs-Fine-Tuning/issues)
- **Model on HuggingFace:** [View model](https://huggingface.co/Antonomics/gemma-2b-financial-qa-lora)
- **Documentation:** [Main README](README.md)

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

**Last Updated:** February 2026  
**Maintained by:** Antony Wahome
