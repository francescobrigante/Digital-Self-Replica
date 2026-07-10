# ==============================================================================
# Tokenization of {prompt, response} pairs with prompt masking (-100 labels),
# plus sanity checks to verify masks and padding are consistent.
# ==============================================================================

import random

from src.tokens import ASSISTANT_TOKEN_START


def tokenize_function(batch, tokenizer, max_length=256):
    """Tokenizes a batch of {prompt, response} pairs for causal LM training.

    Labels equal input_ids with every token before the assistant's response
    masked to -100, so the loss is computed only on the response.
    """
    prompts = batch["prompt"]
    responses = batch["response"]

    tokenized_prompts = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False
    )

    prompt_lengths = [len(tokens) for tokens in tokenized_prompts['input_ids']]  # noqa: F841 (kept for debugging)

    tokenized_conversation = tokenizer(
        [p + r for p, r in zip(prompts, responses)],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        add_special_tokens=False
    )

    input_ids = tokenized_conversation['input_ids']            # full tokenized conversation
    attention_mask = tokenized_conversation['attention_mask']  # real tokens are 1, padding tokens are 0
    labels = []                                                # input_ids with the prompt part masked out

    assistant_token_id = tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN_START)

    for ids in input_ids:
        label = ids.copy()

        # find the last assistant token: the response starts right after it
        response_start_idx = None
        for i in reversed(range(len(label))):
            if label[i] == assistant_token_id:
                response_start_idx = i + 1  # exclude assistant token
                break

        if response_start_idx is None:
            print("[❌] Assistant token not found in input_ids.")
            label = [-100] * len(label)  # ignore everything
        else:
            # mask everything before the assistant's response
            label[:response_start_idx] = [-100] * response_start_idx

        labels.append(label)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def print_tokenized_example(tokenized_dataset, tokenizer, index=None):
    """Prints a random (or indexed) example: ids, decoded text, mask and labels."""
    if index is None:
        index = random.randint(0, len(tokenized_dataset) - 1)
    example = tokenized_dataset[index]

    input_ids = example['input_ids']
    labels = example['labels']
    attention_mask = example['attention_mask']

    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
    decoded_labels = tokenizer.decode(
        [id for id in labels if id != -100],
        skip_special_tokens=False
    )

    print(f"--- Example index: {index} ---")
    print("== INPUT IDS ==")
    print(input_ids)
    print("\n== INPUT TEXT ==")
    print(decoded_input)
    print("\n== ATTENTION MASK ==")
    print(attention_mask)
    print("\n== LABELS (masked prompt) ==")
    print(labels)
    print("\n== LABEL TEXT ==")
    print(decoded_labels)


def verify_tokenized_example(example, tokenizer, assistant_token=ASSISTANT_TOKEN_START, eos_token_id=151643):
    """Verifies two properties of a tokenized example:
    1. the attention mask matches the padding tokens in input_ids;
    2. labels mask exactly the prompt tokens (only the response is not -100).
    """
    input_ids = example['input_ids']
    labels = example['labels']
    attention_mask = example['attention_mask']

    if len(input_ids) != len(labels) or len(input_ids) != len(attention_mask):
        print("[❌] Mismatch in tensor lengths.")
        return False

    # 1: check padding (0s in attention mask == padding token IDs)
    pad_id = tokenizer.pad_token_id or eos_token_id      # EOS used as pad if undefined (standard for deepseek)
    num_padding_tokens = max(0, input_ids.count(pad_id) - 1)  # exclude the last token which is EOS
    num_attention_mask_zeros = attention_mask.count(0)

    if num_padding_tokens != num_attention_mask_zeros:
        print(f"[❌] Padding mismatch: found {num_padding_tokens} pad IDs but {num_attention_mask_zeros} zeros in attention_mask.")
        return False
    print(f"[✅] Attention mask matches padding: {num_padding_tokens}.")

    # 2: visit in reverse to count response tokens (non -100 labels) after the last assistant token
    response_token_count = 0
    for i in reversed(range(len(labels))):
        if labels[i] != -100:
            response_token_count += 1
        elif input_ids[i] == tokenizer.convert_tokens_to_ids(assistant_token):
            break

    actual_response_tokens = sum(1 for l in labels if l != -100)

    if actual_response_tokens != response_token_count:
        print(f"[❌] Label mismatch: expected {response_token_count} response tokens, got {actual_response_tokens}.")
        return False
    print(f"[✅] Labels correctly mask prompt tokens, {response_token_count} response tokens detected.")

    return True


def count_full_attention(dataset):
    """Percentage of examples with no padding — useful to pick the batch size."""
    full_attention_count = 0

    for example in dataset:
        attention_mask = example['attention_mask']
        if all(token == 1 for token in attention_mask):
            full_attention_count += 1

    percentage = (full_attention_count / len(dataset)) * 100
    print(f"✅ Percentage of examples with full attention (no padding): {full_attention_count} / {len(dataset)} = {percentage:.2f}%")
    return percentage
