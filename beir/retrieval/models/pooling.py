from __future__ import annotations

import torch


# CLS pooling useful for encoder based models with CLS Pooling usually used with dot-product similarity.
def cls_pooling(model_output, attention_mask=None):
    return model_output.last_hidden_state[:, 0, :]  # CLS token is first token


# Mean pooling useful for encoder based models with Mean Pooling usually used with cosine similarity and normalization.
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# EOS pooling useful for decoder-only models i.e., large language models like Llama, Mistral, Qwen etc.
def eos_pooling(model_output, attention_mask):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = model_output.last_hidden_state.shape[0]
    return model_output.last_hidden_state[
        torch.arange(batch_size, device=model_output.last_hidden_state.device), sequence_lengths
    ]
