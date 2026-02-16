"""
Data Preprocessing Script for Financial QA Dataset
Converts Financial-QA-10k.csv into Alpaca-formatted JSONL files for fine-tuning
"""

import pandas as pd
import json
import os
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import random
import numpy as np

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Constants
RAW_DATA_PATH = Path("dataset/Financial-QA-10k.csv")
OUTPUT_DIR = Path("data/processed")
MODEL_NAME = "google/gemma-2b"
MAX_SAMPLES = 5000
MAX_SEQ_LENGTH = 2048
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

# Alpaca prompt template
ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def normalize_text(text):
    """
    Normalize text by cleaning whitespace and standardizing formatting.
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string and strip whitespace
    text = str(text).strip()
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Standardize financial notation
    text = re.sub(r'\$\s+', '$', text)  # Remove space after dollar sign
    
    # Remove any control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    
    return text


def create_alpaca_format(row):
    """
    Convert a dataset row into Alpaca format.
    
    Args:
        row: DataFrame row with question, answer, context
        
    Returns:
        Dictionary with instruction, input, output fields
    """
    instruction = normalize_text(row['question'])
    input_context = normalize_text(row['context'])
    output = normalize_text(row['answer'])
    
    return {
        "instruction": instruction,
        "input": input_context,
        "output": output,
        "ticker": row['ticker'],
        "filing": row['filing']
    }


def truncate_context(text, tokenizer, max_tokens=1500):
    """
    Truncate context to fit within token limit while preserving meaning.
    
    Args:
        text: Context text to truncate
        tokenizer: Tokenizer to measure length
        max_tokens: Maximum number of tokens allowed
        
    Returns:
        Truncated text
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate and decode
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    # Try to end at a sentence boundary
    sentences = truncated_text.split('. ')
    if len(sentences) > 1:
        truncated_text = '. '.join(sentences[:-1]) + '.'
    
    return truncated_text


def analyze_sequence_lengths(data, tokenizer):
    """
    Analyze token length distribution of formatted examples.
    
    Args:
        data: List of formatted examples
        tokenizer: Tokenizer to measure length
        
    Returns:
        Dictionary with statistics
    """
    lengths = []
    
    for example in data:
        # Create full prompt
        prompt = ALPACA_TEMPLATE.format(
            instruction=example['instruction'],
            input=example['input'],
            output=example['output']
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        lengths.append(len(tokens))
    
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "p95": np.percentile(lengths, 95),
        "p99": np.percentile(lengths, 99)
    }


def main():
    """Main preprocessing workflow."""
    
    print("=" * 60)
    print("Financial QA Dataset Preprocessing")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load raw dataset
    print(f"\n[1/7] Loading dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   ✓ Loaded {len(df)} examples")
    print(f"   ✓ Companies: {df['ticker'].unique().tolist()}")
    print(f"   ✓ Company distribution:\n{df['ticker'].value_counts().to_string()}")
    
    # Sample data (stratified by company)
    print(f"\n[2/7] Sampling {MAX_SAMPLES} examples (stratified by ticker)...")
    if len(df) > MAX_SAMPLES:
        df_sampled = df.groupby('ticker', group_keys=False).apply(
            lambda x: x.sample(frac=MAX_SAMPLES/len(df), random_state=42)
        ).reset_index(drop=True)
        
        # If we don't have exactly MAX_SAMPLES, adjust
        if len(df_sampled) < MAX_SAMPLES:
            additional = df.drop(df_sampled.index).sample(
                n=MAX_SAMPLES - len(df_sampled), 
                random_state=42
            )
            df_sampled = pd.concat([df_sampled, additional]).reset_index(drop=True)
        elif len(df_sampled) > MAX_SAMPLES:
            df_sampled = df_sampled.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
    else:
        df_sampled = df.copy()
    
    print(f"   ✓ Selected {len(df_sampled)} examples")
    print(f"   ✓ Sampled distribution:\n{df_sampled['ticker'].value_counts().to_string()}")
    
    # Load tokenizer
    print(f"\n[3/7] Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"   ✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    
    # Convert to Alpaca format
    print(f"\n[4/7] Converting to Alpaca format...")
    formatted_data = []
    truncated_count = 0
    
    for idx, row in df_sampled.iterrows():
        example = create_alpaca_format(row)
        
        # Check sequence length and truncate if needed
        full_prompt = ALPACA_TEMPLATE.format(
            instruction=example['instruction'],
            input=example['input'],
            output=example['output']
        )
        token_count = len(tokenizer.encode(full_prompt, add_special_tokens=True))
        
        # Truncate context if exceeds max length
        if token_count > MAX_SEQ_LENGTH:
            # Calculate available tokens for context
            overhead = len(tokenizer.encode(
                ALPACA_TEMPLATE.format(
                    instruction=example['instruction'],
                    input="",
                    output=example['output']
                ),
                add_special_tokens=True
            ))
            
            max_context_tokens = MAX_SEQ_LENGTH - overhead - 50  # Safety margin
            example['input'] = truncate_context(example['input'], tokenizer, max_context_tokens)
            truncated_count += 1
        
        formatted_data.append(example)
    
    print(f"   ✓ Formatted {len(formatted_data)} examples")
    print(f"   ✓ Truncated {truncated_count} contexts to fit within {MAX_SEQ_LENGTH} tokens")
    
    # Analyze sequence lengths
    print(f"\n[5/7] Analyzing sequence lengths...")
    stats = analyze_sequence_lengths(formatted_data, tokenizer)
    print(f"   ✓ Token length statistics:")
    print(f"      - Min: {stats['min']}")
    print(f"      - Max: {stats['max']}")
    print(f"      - Mean: {stats['mean']:.1f}")
    print(f"      - Median: {stats['median']:.1f}")
    print(f"      - 95th percentile: {stats['p95']:.1f}")
    print(f"      - 99th percentile: {stats['p99']:.1f}")
    
    # Split data
    print(f"\n[6/7] Splitting data (train: {TRAIN_RATIO:.0%}, val: {VAL_RATIO:.0%}, test: {TEST_RATIO:.0%})...")
    
    # First split: train + (val + test)
    train_data, temp_data = train_test_split(
        formatted_data,
        train_size=TRAIN_RATIO,
        random_state=42,
        stratify=[d['ticker'] for d in formatted_data]
    )
    
    # Second split: val and test
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_ratio_adjusted,
        random_state=42,
        stratify=[d['ticker'] for d in temp_data]
    )
    
    print(f"   ✓ Train: {len(train_data)} examples")
    print(f"   ✓ Validation: {len(val_data)} examples")
    print(f"   ✓ Test: {len(test_data)} examples")
    
    # Save to JSONL files
    print(f"\n[7/7] Saving JSONL files to {OUTPUT_DIR}...")
    
    def save_jsonl(data, filename):
        """Save data to JSONL file."""
        filepath = OUTPUT_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in data:
                # Remove metadata fields for training
                training_example = {
                    "instruction": example['instruction'],
                    "input": example['input'],
                    "output": example['output']
                }
                f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
        print(f"   ✓ Saved {filename} ({len(data)} examples)")
        return filepath
    
    train_file = save_jsonl(train_data, "train.jsonl")
    val_file = save_jsonl(val_data, "validation.jsonl")
    test_file = save_jsonl(test_data, "test.jsonl")
    
    # Save metadata with ticker information
    metadata = {
        "total_samples": len(formatted_data),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "max_seq_length": MAX_SEQ_LENGTH,
        "model_name": MODEL_NAME,
        "truncated_contexts": truncated_count,
        "token_stats": stats,
        "ticker_distribution": df_sampled['ticker'].value_counts().to_dict()
    }
    
    metadata_file = OUTPUT_DIR / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved metadata.json")
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {test_file}")
    print(f"  - {metadata_file}")
    print("\nReady for training!")


if __name__ == "__main__":
    main()
