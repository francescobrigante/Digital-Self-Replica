# Digital Self-Replica: Personal Style Cloning via QLoRA

This project implements a Personal Style Cloning system that adapts a Large Language Model to mimic an individual's communication patterns using Quantized Low-Rank Adaptation (QLoRA). The idea is to fine-tune a baseline LLM on personal WhatsApp conversation data to create a Digital Replica capable of generating responses in the target individual's characteristic writing style.

## Project Overview

The system uses QLoRA (Quantized Low-Rank Adaptation) to efficiently fine-tune a 7B parameter model (DeepSeek-R1-Distill-Qwen-7B) on personal messaging data. By combining 4-bit quantization with Low-Rank Adaptation, the approach enables training large language models on consumer-grade hardware while maintaining good performances.

## Repository Structure
```
├── chat/                       # Raw WhatsApp chats (empty - private data)
├── data/                       # Pre-processed CSV files (empty - private data)
├── datasets/                   # Final datasets (empty - private data)
├── francesco_lora/             # Checkpoint directory (empty - private data)
├── data.ipynb                  # Data preprocessing and dataset creation logic
├── evaluation.ipynb            # Model evaluation and metrics computation
├── lora_train.ipynb            # QLoRA fine-tuning implementation
├── prompt_testing.ipynb        # Interactive evaluation and response testing
├── .DS_Store
├── .gitignore
└── LICENSE
```


## File Descriptions

### `data.ipynb`

- WhatsApp chat data preprocessing and cleaning  
- Dataset creation with prompt-response pairs  
- Context window implementation (sliding window of up to 5 messages)  
- Special token integration for role identification  
- Train/validation/test split (80%/10%/10%)
- Utility functions to check some statistics of the computed dataset

### `lora_train.ipynb`

- QLoRA configuration and setup  
- Model loading with 4-bit quantization  
- Training loop with evaluation and checkpointing  
- Weights & Biases integration for experiment tracking

### `evaluation.ipynb`

- Comprehensive model evaluation using multiple metrics:
  - Perplexity computation  
  - METEOR score for lexical similarity  
  - BERTScore F1 for semantic evaluation  
  - Semantic similarity using sentence embeddings  
  - Custom prompt alignment ratio  
- Baseline vs fine-tuned model comparison
- Generated vs Ground truth response comparison


### `prompt_testing.ipynb`

- Interactive evaluation with direct prompting  
- Qualitative assessment of generated responses  
- Style consistency and response diversity analysis
- Zero-shot domain generalization experiments  

## Privacy Notice

**Important:** The following directories are intentionally empty as they contain private conversational data:

- `chat/` - Raw WhatsApp conversation files  
- `data/` - Processed conversation datasets  
- `datasets/` - Training/validation/test splits  
- `francesco_lora/` - Trained model adapters  

The private dataset consists of approximately 65,000 WhatsApp messages spanning from 2020, resulting in ~14,000 examples after preprocessing.

## Technical Details

### Model Architecture

- **Base Model:** DeepSeek-R1-Distill-Qwen-7B (7B parameters)  
- **Quantization:** 4-bit precision using BitsAndBytes  
- **Adaptation:** LoRA with rank=64, alpha=32 (chosen from the best model)  
- **Trainable Parameters:** ~160M (2% of original model)

### Training Configuration

- **Optimizer:** Paged AdamW 8-bit  
- **Learning Rate:** 2×10⁻⁴ with cosine scheduling  
- **Batch Processing:** Gradient accumulation for memory efficiency  
- **Early Stopping:** Based on evaluation loss  
- **Training epochs:** 3 (training split across 2 runs)

## Key Results

| Metric               | Baseline | Fine-Tuned | Improvement        |
|----------------------|----------|------------|--------------------|
| Perplexity           | 9,286.86 | 17.99      | -99.8%             |
| BERTScore F1         | 0.275    | 0.414      | +50%               |
| Semantic Similarity  | 0.22     | 0.285      | +30%               |
| Prompt Alignment     | 260%     | 104%       | Near-optimal       |
