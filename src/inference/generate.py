# ==============================================================================
# Chat inference: prompt formatting with optional conversation history
# and sampling-based generation with the project's decoding settings.
# ==============================================================================

import torch

from src.prompts import CHAT_SYSTEM_PROMPT
from src.tokens import get_special_tokens


def format_prompt(new_user_query, tokenizer, conversation_history=None,
                  system_prompt=CHAT_SYSTEM_PROMPT):
    """Formats a chat prompt: system prompt, optional history, then the new query."""
    tokens = get_special_tokens(tokenizer)
    conversation_history = conversation_history or []

    prompt_parts = []

    if system_prompt:
        prompt_parts.append(f"{tokens.bos}{system_prompt}")

    for turn in conversation_history:
        role = turn.get("role")
        content = turn.get("content")
        if role == "user":
            prompt_parts.append(f"{tokens.user_start}{content}{tokens.end_turn}")
        elif role == "assistant":
            prompt_parts.append(f"{tokens.assistant_start}{content}{tokens.end_turn}")

    prompt_parts.append(f"{tokens.user_start}{new_user_query}{tokens.end_turn}")
    prompt_parts.append(tokens.assistant_start)

    return "\n".join(prompt_parts)


def generate(user_input, model, tokenizer, conversation_history=None,
             max_new_tokens=90, temperature=0.4, top_p=0.95,
             repetition_penalty=1.2, device=None, verbose=True):
    """Generates a reply in Francesco's style; returns the cleaned response text."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokens = get_special_tokens(tokenizer)
    prompt = format_prompt(user_input, tokenizer, conversation_history)

    if verbose:
        print("\n-------- Prompt -----------------")
        print(prompt)
        print("---------------------------------\n")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,   # stop generation when this token is produced
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=1,
        )

    generated_ids = outputs[0, input_ids_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # clean up response removing everything after EOS
    if tokens.eos in generated_text:
        cleaned_response = generated_text.split(tokens.eos)[0].strip()
    else:
        cleaned_response = generated_text.strip()

    if verbose:
        print("\n--- Response ---")
        print(cleaned_response)
        print("-----------------------\n")

    return cleaned_response
