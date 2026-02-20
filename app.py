"""
Financial Assistant - Streamlit Web Interface
Fine-tuned Gemma-2B model for financial question answering
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import time
import os

# Page configuration
st.set_page_config(
    page_title="Financial Assistant AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for finance-themed interface
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 2px solid #2e7d32;
    }
    .stTextArea > div > div > textarea {
        background-color: #ffffff;
        border: 2px solid #2e7d32;
    }
    .financial-header {
        background: linear-gradient(90deg, #1b5e20 0%, #2e7d32 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .response-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
        margin: 10px 0;
    }
    .context-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f57c00;
        margin: 10px 0;
    }
    .stButton > button {
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 30px;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1b5e20;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="financial-header">
        <h1>Financial Assistant AI</h1>
        <p>Ask questions about financial reports, SEC filings, and company performance</p>
    </div>
""", unsafe_allow_html=True)

# Model configuration
MODEL_ID = "Antonomics/gemma-2b-financial-qa-lora"

@st.cache_resource
def load_model():
    """Load the fine-tuned model from HuggingFace"""
    with st.spinner(" Loading AI model... This may take a minute..."):
        try:
            # Use stored HuggingFace credentials (set via huggingface-hub login)
            # Token will be automatically loaded from ~/.huggingface/token
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Configure quantization for efficient inference
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load base model
            base_model_name = "google/gemma-2b"
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Load LoRA adapters from the fine-tuned model
            model = PeftModel.from_pretrained(
                model, 
                MODEL_ID
            )
            model.eval()
            
            return tokenizer, model
        except Exception as e:
            error_msg = str(e)
            st.error(f"**Error loading model:** {error_msg}")
            
            if "gated" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                st.warning("""
                    **Troubleshooting Authentication:**
                    
                    1. Verify you accepted the license at https://huggingface.co/google/gemma-2b
                    2. Try re-running the login command:
                       ```python
                       .venv\\Scripts\\python.exe -c "from huggingface_hub import login; login()"
                       ```
                    3. Check your token permissions at https://huggingface.co/settings/tokens
                    4. Clear cache and reload (Press 'C' in the app or restart Streamlit)
                """)
            
            return None, None

def generate_response(tokenizer, model, instruction, input_context="", max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Generate response from the model"""
    # Format prompt using Alpaca template
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_context if input_context else "No additional context provided."}

### Response:
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    response = full_output.split("### Response:")[-1].strip()
    
    return response

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    
    # Generation parameters
    st.markdown("#### Generation Parameters")
    max_tokens = st.slider("Max Response Length", 50, 500, 256, 
                          help="Maximum number of tokens to generate")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1,
                           help="Higher values make output more creative, lower values more focused")
    top_p = st.slider("Top-p (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.05,
                     help="Controls diversity of output")
    
    st.markdown("---")
    
    # Clear cache button
    if st.button("🔄 Reload Model", help="Clear cache and reload the model"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Example questions
    st.markdown("###  Example Questions")
    examples = [
        "What was the company's total revenue?",
        "What were the main risk factors?",
        "How did the company perform compared to last year?",
        "What are the key business segments?",
        "What is the company's debt situation?",
    ]
    
    for example in examples:
        if st.button(example, key=example):
            st.session_state.example_question = example
    
    st.markdown("---")
    
    # Model info
    with st.expander(" Model Information"):
        st.markdown(f"""
        **Base Model:** Google Gemma-2B  
        **Fine-tuning Method:** QLoRA (4-bit)  
        **Dataset:** Financial-QA-10K  
        **Training Examples:** 3,000+  
        **HuggingFace:** [View Model]({MODEL_ID})
        """)
    
    # Clear history
    if st.button(" Clear History"):
        st.session_state.history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("###  Ask Your Financial Question")
    
    # Check if example was clicked
    default_question = st.session_state.get('example_question', '')
    if default_question:
        question = st.text_input(
            "Enter your question:",
            value=default_question,
            placeholder="e.g., What was the company's revenue growth?",
            help="Ask any question about financial reports or SEC filings"
        )
        st.session_state.example_question = ''  # Reset after use
    else:
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What was the company's revenue growth?",
            help="Ask any question about financial reports or SEC filings"
        )
    
    context = st.text_area(
        "Context (Optional):",
        placeholder="Paste relevant excerpts from financial reports, 10-K filings, or other financial documents here...",
        height=150,
        help="Provide context from financial documents to get more accurate answers"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        submit = st.button(" Get Answer", type="primary")
    with col_btn2:
        clear_inputs = st.button(" Clear")

with col2:
    st.markdown("###  Quick Stats")
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color: #2e7d32; margin: 0;">Questions Asked</h4>
        <h2 style="margin: 5px 0;">{len(st.session_state.history)}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #2e7d32; margin: 0;">Model Status</h4>
        <h3 style="margin: 5px 0; color: #1b5e20;">✓ Ready</h3>
    </div>
    """, unsafe_allow_html=True)

# Handle clear button
if clear_inputs:
    st.rerun()

# Handle question submission
if submit and question:
    # Load model
    tokenizer, model = load_model()
    
    if tokenizer and model:
        with st.spinner(" Analyzing your question..."):
            start_time = time.time()
            
            # Generate response
            response = generate_response(
                tokenizer, 
                model, 
                question, 
                context,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            end_time = time.time()
            response_time = end_time - start_time
        
        # Display response
        st.markdown("###  AI Assistant Response")
        st.markdown(f"""
        <div class="response-box">
            <p style="margin: 0; font-size: 16px;">{response}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"⏱ Response generated in {response_time:.2f} seconds")
        
        # Add to history
        st.session_state.history.append({
            'question': question,
            'context': context,
            'response': response,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Show context if provided
        if context:
            with st.expander(" Context Used"):
                st.markdown(f"""
                <div class="context-box">
                    <p style="margin: 0; font-size: 14px;">{context[:500]}{"..." if len(context) > 500 else ""}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error(" Failed to load model. Please try again.")

elif submit and not question:
    st.warning(" Please enter a question before submitting.")

# Display history
if st.session_state.history:
    st.markdown("---")
    st.markdown("###  Question History")
    
    for i, item in enumerate(reversed(st.session_state.history[-5:])):
        with st.expander(f"Q{len(st.session_state.history) - i}: {item['question'][:60]}... - {item['timestamp']}"):
            st.markdown(f"**Question:** {item['question']}")
            if item['context']:
                st.markdown(f"**Context:** {item['context'][:200]}...")
            st.markdown(f"""
            <div class="response-box">
                <strong>Response:</strong><br>
                {item['response']}
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p> Powered by Gemma-2B fine-tuned on Financial-QA-10K dataset</p>
        <p style="font-size: 12px;">Built with  using Streamlit | Model on <a href="https://huggingface.co/Antonomics/gemma-2b-financial-qa-lora" target="_blank">HuggingFace</a></p>
    </div>
""", unsafe_allow_html=True)
