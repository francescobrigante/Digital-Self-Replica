# ==============================================================================
# CLI: interactive chat with the fine-tuned replica.
# Usage: python scripts/chat.py --adapter-path francesco_lora/checkpoint-600
# Type /reset to clear the conversation history, /quit to exit.
# On constrained GPUs pass e.g. --gpu-mem 15GiB --cpu-mem 28GiB --offload-dir ./offload
# ==============================================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.inference.generate import generate
from src.model_loading import load_finetuned_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Chat with the fine-tuned digital replica.")
    parser.add_argument("--config", default="config.yaml", help="path to the config file")
    parser.add_argument("--adapter-path", required=True, help="path to the trained LoRA adapter checkpoint")
    parser.add_argument("--offload-dir", default=None, help="offload dir for layers exceeding the memory budget")
    parser.add_argument("--gpu-mem", default=None, help='GPU memory budget, e.g. "15GiB"')
    parser.add_argument("--cpu-mem", default=None, help='CPU memory budget, e.g. "28GiB"')
    parser.add_argument("--history-turns", type=int, default=10,
                        help="max conversation turns kept as context (0 disables history)")
    parser.add_argument("--show-prompt", action="store_true", help="print the full formatted prompt at each turn")
    args = parser.parse_args()

    cfg = load_config(args.config)

    max_memory = None
    if args.gpu_mem or args.cpu_mem:
        max_memory = {}
        if args.gpu_mem:
            max_memory[0] = args.gpu_mem
        if args.cpu_mem:
            max_memory["cpu"] = args.cpu_mem

    print(f"Loading {cfg.model.base_model_id} + adapter {args.adapter_path} ...")
    tokenizer = load_tokenizer(cfg.model.base_model_id)
    model = load_finetuned_model(
        cfg.model.base_model_id,
        args.adapter_path,
        offload_dir=args.offload_dir,
        max_memory=max_memory,
    )

    print("Ready. /reset clears the history, /quit exits.\n")
    history = []

    while True:
        try:
            user_input = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/reset":
            history = []
            print("(history cleared)\n")
            continue

        reply = generate(
            user_input,
            model=model,
            tokenizer=tokenizer,
            conversation_history=history,
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            top_p=cfg.generation.top_p,
            repetition_penalty=cfg.generation.repetition_penalty,
            verbose=args.show_prompt,
        )
        print(f"Francesco: {reply}\n")

        if args.history_turns > 0:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})
            history = history[-2 * args.history_turns:]  # keep the last N turns


if __name__ == "__main__":
    main()
