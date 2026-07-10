# ==============================================================================
# CLI: QLoRA fine-tuning of the base model on the tokenized dataset.
# Usage: python scripts/train.py [--config config.yaml] [--wandb] [--resume-from CKPT]
# Secrets (HF_TOKEN, WANDB_API_KEY) are read from the environment / .env file.
# ==============================================================================

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_from_disk
from dotenv import load_dotenv
from huggingface_hub import login

from src.config import load_config
from src.model_loading import load_base_model
from src.training.lora_trainer import build_trainer, prepare_lora_model


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning on the tokenized WhatsApp dataset.")
    parser.add_argument("--config", default="config.yaml", help="path to the config file")
    parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--run-name", default=None, help="W&B run name")
    parser.add_argument("--resume-from", default=None, help="checkpoint path to resume training from")
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        sys.exit("HF_TOKEN is not set (put it in your environment or in a .env file).")
    login(token=hf_token)

    cfg = load_config(args.config)

    if args.wandb:
        import wandb
        wandb.login()  # uses WANDB_API_KEY from the environment
        wandb.init(project=cfg.training.wandb_project, job_type="Training", name=args.run_name)

    datasets_dir = Path(cfg.data.datasets_dir)
    tokenized_train = load_from_disk(str(datasets_dir / "tokenized_train"))
    tokenized_val = load_from_disk(str(datasets_dir / "tokenized_val"))

    model = load_base_model(cfg.model.base_model_id)
    model = prepare_lora_model(model, cfg.lora)
    model.print_trainable_parameters()

    trainer = build_trainer(model, cfg.training, tokenized_train, tokenized_val, use_wandb=args.wandb)
    trainer.train(resume_from_checkpoint=args.resume_from)


if __name__ == "__main__":
    main()
