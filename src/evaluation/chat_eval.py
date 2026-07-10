# ==============================================================================
# Generation-quality evaluation: METEOR, BERTScore (Italian BERT) and
# sentence-embedding semantic similarity between predictions, references
# and prompts. Predictions are collected first, metrics computed in one shot.
# ==============================================================================

import numpy as np
import torch
from evaluate import load
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

ITALIAN_SBERT_MODEL = 'nickprock/sentence-bert-base-italian-uncased'
ITALIAN_BERTSCORE_MODEL = 'dbmdz/bert-base-italian-xxl-cased'


def left_pad(sequences, pad_value):
    """Pads a list of 1D tensors to the same length on the left."""
    max_len = max(seq.size(0) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padded_seq = torch.cat([torch.full((pad_len,), pad_value, dtype=seq.dtype, device=seq.device), seq])
        padded.append(padded_seq)
    return torch.stack(padded)                                # (B, max_len)


def convert_label_to_string(label, tokenizer, skip_special_tokens=True):
    """Decodes a label tensor into text, skipping the -100 masked positions."""
    valid_token_ids = label[label != -100]
    return tokenizer.decode(valid_token_ids.tolist(), skip_special_tokens=skip_special_tokens)


def print_batch_debug(batch_prompts, responses, ground_truths, tokenizer, N=3):
    """Prints up to N (prompt, generated, ground truth) triples of a batch."""
    to_print = min(N, len(batch_prompts))
    for idx in range(to_print):
        prompt_txt = tokenizer.decode(batch_prompts[idx].tolist(), skip_special_tokens=True)
        gen_txt = tokenizer.decode(responses[idx].tolist(), skip_special_tokens=True)
        gt_txt = ground_truths[idx]

        print(f"{'-'*10} Example {idx+1} {'-'*10}")
        print(f"Prompt:\n{prompt_txt}")
        print(f"\nGenerated:    {gen_txt}")
        print(f"Ground Truth: {gt_txt}")
        print()


def extract_prompts_from_batch(input_ids, labels):
    """Recovers per-example prompt token sequences using the -100 label mask."""
    batch_prompts = []
    for i in range(input_ids.size(0)):
        prompt_tokens = input_ids[i][labels[i] == -100]
        batch_prompts.append(prompt_tokens)
    return batch_prompts


def generate_batch_responses(model, padded_prompts, pad_token_id,
                             max_new_tokens=90, temperature=0.4, top_p=0.95):
    """Samples one response per prompt; returns only the generated part."""
    generated = model.generate(
        input_ids=padded_prompts,
        attention_mask=(padded_prompts != pad_token_id).long(),
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=pad_token_id,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        return_dict_in_generate=True,
        output_scores=False,
    )
    return generated.sequences[:, padded_prompts.size(1):]   # (B, new_tokens)


def analyze_generation(model, dataloader, tokenizer, device="cuda", print_every=10):
    """Qualitative pass: periodically prints prompts, generations and ground truths."""
    model.eval()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generation: ")):
        input_ids = batch['input_ids'].to(device)             # (B, seq_len)
        labels = batch['labels'].to(device)                   # (B, seq_len)

        ground_truths = [convert_label_to_string(label, tokenizer) for label in labels]

        pad_token_id = tokenizer.pad_token_id
        batch_prompts = extract_prompts_from_batch(input_ids, labels)
        padded_prompts = left_pad(batch_prompts, pad_token_id).to(device)

        with torch.no_grad():
            responses = generate_batch_responses(model, padded_prompts, pad_token_id)

        if (batch_idx + 1) % print_every == 0:
            print_batch_debug(padded_prompts, responses, ground_truths, tokenizer, N=3)


def evaluate_chat_model(model, tokenizer, dataloader, device=None):
    """Full quantitative evaluation of a chat model.

    First loops to generate the predictions, then measures everything in one
    shot to speed up the process. Returns METEOR, BERTScore F1, semantic
    similarity and the prompt-alignment ratio (how close predictions stay to
    the prompt compared to how close the ground truth does).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    meteor = load("meteor")
    bertscore = load("bertscore")

    semantic_model = SentenceTransformer(
        ITALIAN_SBERT_MODEL,
        device='cpu'    # cuda already full
    )

    all_prompts: list[str] = []
    all_preds: list[str] = []
    all_references: list[str] = []

    pad_token_id = tokenizer.pad_token_id

    # generation loop: collecting predictions
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating & Collecting"):
            input_ids = batch['input_ids'].to(device)         # (B, seq_len)
            labels = batch['labels'].to(device)               # (B, seq_len)

            batch_prompts_ids = extract_prompts_from_batch(input_ids, labels)
            padded_prompts = left_pad(batch_prompts_ids, pad_token_id).to(device)

            all_prompts.extend(tokenizer.batch_decode(padded_prompts, skip_special_tokens=True))
            all_references.extend(convert_label_to_string(label, tokenizer) for label in labels)

            gen_tokens = generate_batch_responses(model, padded_prompts, pad_token_id)
            all_preds.extend(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))

    # metrics computed on the collected lists

    meteor_res = meteor.compute(predictions=all_preds, references=all_references)
    avg_meteor = meteor_res["meteor"]

    bert_res = bertscore.compute(
        predictions=all_preds,
        references=all_references,
        lang="it",
        model_type=ITALIAN_BERTSCORE_MODEL,
        num_layers=12,
        device="cpu",
        batch_size=16,
        rescale_with_baseline=False,
    )
    avg_bertscore_f1 = float(np.mean(bert_res["f1"]))

    encode_kwargs = dict(convert_to_tensor=True, normalize_embeddings=True,
                         batch_size=64, show_progress_bar=True)
    prompt_embeds = semantic_model.encode(all_prompts, **encode_kwargs)
    pred_embeds = semantic_model.encode(all_preds, **encode_kwargs)
    ref_embeds = semantic_model.encode(all_references, **encode_kwargs)

    sim_pred_ref = util.cos_sim(pred_embeds, ref_embeds).diagonal().cpu().numpy()
    sim_prompt_pred = util.cos_sim(prompt_embeds, pred_embeds).diagonal().cpu().numpy()
    sim_prompt_ref = util.cos_sim(prompt_embeds, ref_embeds).diagonal().cpu().numpy()

    avg_semantic_similarity = float(np.mean(sim_pred_ref))
    avg_prompt_pred = float(np.mean(sim_prompt_pred))
    avg_prompt_ref = float(np.mean(sim_prompt_ref))
    prompt_alignment_ratio = avg_prompt_pred / (avg_prompt_ref + 1e-12)

    return {
        "meteor": avg_meteor,
        "bertscore_f1": avg_bertscore_f1,
        "semantic_similarity": avg_semantic_similarity,
        "prompt_alignment": {
            "predicted": avg_prompt_pred,
            "ground_truth": avg_prompt_ref,
            "ratio": prompt_alignment_ratio,
        },
        "predictions": all_preds,
        "references": all_references,
    }


def print_evaluation_results(results, model_name):
    """Pretty-prints the dictionary returned by evaluate_chat_model."""
    print(f"""
    {model_name} Evaluation Results:
    - METEOR: {results['meteor']:.3f} (0-1, higher=better)
    - BERTScore F1: {results['bertscore_f1']:.3f} (0-1, higher=better)
    - Semantic Similarity: {results['semantic_similarity']:.3f} (0-1 cosine)
    Prompt Alignment:
    - Model Responses: {results['prompt_alignment']['predicted']:.3f}
    - Ground Truth: {results['prompt_alignment']['ground_truth']:.3f}
    - Alignment Ratio: {results['prompt_alignment']['ratio']:.1%}
""")
