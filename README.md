# Digital Self-Replica: Personal Style Cloning via QLoRA

This project implements a Personal Style Cloning system that adapts a Large Language Model to mimic an individual's communication patterns using Quantized Low-Rank Adaptation (QLoRA). The idea is to fine-tune a baseline LLM on personal WhatsApp conversation data to create a Digital Replica capable of generating responses in the target individual's characteristic writing style.

For further details, read the full paper [NLP HWp Francesco Brigante.pdf](/NLP%20HWp%20Francesco%20Brigante.pdf)

## Project Overview

The system uses QLoRA (Quantized Low-Rank Adaptation) to efficiently fine-tune a 7B parameter model (DeepSeek-R1-Distill-Qwen-7B) on personal messaging data. By combining 4-bit quantization with Low-Rank Adaptation, the approach enables training large language models on consumer-grade hardware while maintaining good performances.

**Pipeline:** WhatsApp exports → cleaning & filtering → prompt/response pairs with time-aware context → tokenization with prompt masking → QLoRA fine-tuning → quantitative + qualitative evaluation → chat inference.

## Repository Structure

The pipeline logic lives in the `src/` package; the notebooks are thin showcases that import from it and preserve the original training/evaluation outputs.

```
├── config.yaml                    # All hyperparameters (data, LoRA, training, generation)
├── requirements.txt
├── src/
│   ├── config.py                  # Typed access to config.yaml
│   ├── tokens.py                  # DeepSeek special tokens (shared by data/inference/eval)
│   ├── prompts.py                 # Training and chat system prompts
│   ├── model_loading.py           # 4-bit quantized model loading, with/without LoRA adapter
│   ├── data/
│   │   ├── whatsapp_parser.py     # WhatsApp .txt → cleaned CSV
│   │   ├── dataset_builder.py     # CSV → conversations JSON → prompt/response pairs
│   │   └── tokenization.py        # Tokenization with prompt masking + sanity checks
│   ├── training/
│   │   └── lora_trainer.py        # LoRA config, TrainingArguments, Trainer assembly
│   ├── inference/
│   │   └── generate.py            # Prompt formatting and chat generation
│   └── evaluation/
│       ├── perplexity.py          # Token-weighted perplexity
│       └── chat_eval.py           # METEOR, BERTScore, semantic similarity, prompt alignment
├── scripts/
│   ├── build_dataset.py           # CLI: chats → tokenized train/val/test datasets
│   └── train.py                   # CLI: QLoRA fine-tuning
├── notebooks/
│   ├── 01_data_pipeline.ipynb     # Dataset construction walkthrough with real examples
│   ├── 02_lora_training.ipynb     # Training run record (Kaggle) with loss curves
│   ├── 03_evaluation.ipynb        # Baseline vs fine-tuned metrics across LoRA ranks
│   └── 04_chat_demo.ipynb         # Live conversations with the replica
├── chat/                          # Raw WhatsApp chats (empty - private data)
├── data/                          # Pre-processed CSV files (empty - private data)
├── datasets/                      # Final datasets (empty - private data)
└── francesco_lora/                # Checkpoint directory (empty - private data)
```

## Quick Start

```bash
pip install -r requirements.txt

# put your HF token in .env (HF_TOKEN=...), export WhatsApp chats as .txt into chat/
python scripts/build_dataset.py            # chats → tokenized datasets in datasets/
python scripts/train.py --wandb            # QLoRA fine-tuning (needs a GPU)
```

Every hyperparameter (context window, time gap, LoRA rank/alpha, training arguments, decoding settings) is in [config.yaml](config.yaml) — nothing is hardcoded in the source.

## Notebooks

### `01_data_pipeline.ipynb`

- WhatsApp chat data preprocessing and cleaning
- Dataset creation with prompt-response pairs
- Context window implementation (time-gap heuristic, up to 5 messages)
- Special token integration for role identification
- Train/validation/test split (80%/10%/10%)
- Sanity checks: prompt masking, padding/attention consistency, dataset statistics

### `02_lora_training.ipynb`

- QLoRA configuration and setup
- Model loading with 4-bit quantization
- Training loop with evaluation and checkpointing
- Weights & Biases integration for experiment tracking

### `03_evaluation.ipynb`

- Comprehensive model evaluation using multiple metrics:
  - Perplexity computation
  - METEOR score for lexical similarity
  - BERTScore F1 for semantic evaluation
  - Semantic similarity using sentence embeddings
  - Custom prompt alignment ratio
- Baseline vs fine-tuned model comparison across LoRA ranks (r=16/32/64)
- Generated vs Ground truth response comparison

### `04_chat_demo.ipynb`

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
- **Learning Rate:** 3×10⁻⁴ with cosine scheduling
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
