# ==============================================================================
# Perplexity of a (fine-tuned or base) model over the tokenized test set.
# The loss is token-weighted so batches with more response tokens count more.
# ==============================================================================

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ChatDataset(Dataset):
    """Wraps a tokenized HF dataset into tensors for a PyTorch DataLoader."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.encodings["labels"][idx], dtype=torch.long),
        }


def calculate_perplexity(model, dataloader, device="cuda"):
    """Computes exp(mean loss per response token) over the dataloader."""
    model = model.to(device)
    model.eval()
    total_loss = 0
    total_tokens_in_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Perplexity"):
            input_ids = batch['input_ids'].to(device)          # (B, seq_len)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            num_tokens = (labels != -100).sum().item()
            if num_tokens == 0:
                continue

            # weight the batch loss by its number of response tokens
            total_loss += loss.item() * num_tokens
            total_tokens_in_loss += num_tokens

    avg_loss = total_loss / total_tokens_in_loss
    perplexity = torch.exp(torch.tensor(avg_loss, device=device))
    return perplexity.item()
