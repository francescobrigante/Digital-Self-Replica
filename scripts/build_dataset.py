# ==============================================================================
# CLI: WhatsApp .txt exports → cleaned CSVs → conversations JSON →
# tokenized HF train/val/test datasets saved to disk.
# Usage: python scripts/build_dataset.py [--config config.yaml]
# ==============================================================================

import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.dataset_builder import build_splits, create_dataset, create_dataset_list
from src.data.tokenization import tokenize_function
from src.data.whatsapp_parser import process_data
from src.model_loading import load_tokenizer
from src.tokens import get_special_tokens


def main():
    parser = argparse.ArgumentParser(description="Build the tokenized dataset from WhatsApp chat exports.")
    parser.add_argument("--config", default="config.yaml", help="path to the config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = load_tokenizer(cfg.model.base_model_id)
    tokens = get_special_tokens(tokenizer)

    # 1. WhatsApp .txt → cleaned CSV, one per chat in chat_dir
    chat_files = sorted(Path(cfg.data.chat_dir).glob("*.txt"))
    if not chat_files:
        sys.exit(f"No .txt chats found in '{cfg.data.chat_dir}/'. Export your WhatsApp chats there first.")

    csv_files = []
    for chat_file in chat_files:
        csv_path = Path(cfg.data.csv_dir) / f"{chat_file.stem}.csv"
        process_data(str(chat_file), str(csv_path), assistant_name=cfg.data.assistant_name)
        csv_files.append(str(csv_path))

    # 2. CSVs → merged conversations JSON
    create_dataset(csv_files, cfg.data.dataset_json, assistant_name=cfg.data.assistant_name)

    # 3. JSON → {prompt, response} examples
    dataset_list = create_dataset_list(
        cfg.data.dataset_json,
        tokens=tokens,
        max_context_messages=cfg.data.max_context_messages,
        time_gap_seconds=cfg.data.time_gap_seconds,
    )
    print(f"Generated {len(dataset_list)} training examples.")

    # 4. splits + tokenization with prompt masking
    train_dataset, val_dataset, test_dataset = build_splits(
        dataset_list, test_size=cfg.data.test_size, seed=cfg.data.seed
    )

    tokenize = partial(tokenize_function, tokenizer=tokenizer, max_length=cfg.data.max_length)
    datasets_dir = Path(cfg.data.datasets_dir)
    for name, split in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        tokenized = split.map(tokenize, batched=True, remove_columns=split.column_names)
        tokenized.save_to_disk(str(datasets_dir / f"tokenized_{name}"))
        print(f"{name}: {len(tokenized)} examples → {datasets_dir / f'tokenized_{name}'}")


if __name__ == "__main__":
    main()
