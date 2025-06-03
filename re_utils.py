from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
import torch
import logging
import pathlib, os

# critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F


#### Provide the data_path where scifact has been downloaded and unzipped

import torch

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
import torch

class CriticModel(torch.nn.Module):
    def __init__(self):

        pass
    
    def forward():
        pass

def llm_output(llm_model, tokenizer, queries, query_facts, 
               max_length=256,
               batch_size=None, **generate_kwargs):
    """
    Perform batch inference with a causal LLM, combining each query with its associated facts.

    Inputs:
        llm_model:      An accelerator-wrapped AutoModelForCausalLM already on the correct device.
        tokenizer:      The corresponding AutoTokenizer (not wrapped by accelerator).
        queries:        List[str] of length B, each a user query.
        query_facts:    List[List[str]] of length B, where query_facts[i] is a list of fact-strings for queries[i].
        max_length:     Maximum total generation length (including prompt).
        batch_size:     Optional int. If provided, processes inputs in chunks of this size; 
                        otherwise, all B inputs are processed in one batch.
        **generate_kwargs: Additional keyword arguments to pass to llm_model.generate().

    Returns:
        actions: List[str] of length B, where each element is the decoded output for the corresponding query+facts.
    """
    device = llm_model.device
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Build full prompts by concatenating each query with its facts
    prompts = []
    for q, facts in zip(queries, query_facts):
        # Example prompt template: you can adjust this to your preferred formatting
        # Here, we prefix with "Question:" and list facts under "Facts:"
        fact_section = ""
        if facts:
            fact_section = "Facts:\n" + "\n".join(f"- {f}" for f in facts) + "\n"
        prompt = f"Question: {q}\n{fact_section}Answer: (Please choose from 'A, B, C, D' and answer with the letter at the very beginning or very end of you response.)"
        prompts.append(prompt)

    # 2. Decide batch size
    B = len(prompts)
    if batch_size is None or batch_size >= B:
        batch_size = B

    all_outputs = []
    llm_model.eval()
    with torch.no_grad():
        # 3. Process in chunks of size batch_size
        for i in range(0, B, batch_size):
            chunk_prompts = prompts[i : i + batch_size]

            # 3a. Tokenize the chunk of prompts
            encoding = tokenizer(
                chunk_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # 3b. Generate output IDs
            generated_ids = llm_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
                **generate_kwargs,
            )

            # 3c. Decode each generated sequence (skip the prompt tokens if desired)
            #    Here, we decode the entire generated_ids and return full text.
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_outputs.extend(decoded)

    return all_outputs

def llm_reward_computing(actions, gt_answers, gt_paragraphs=None):

    '''
    Handles reward computing. 

    Inputs:
        * actions: B x words, LLM output
        * gt_answer: B x words, ground truth output for LLM
    
    Outputs:
        * rewards: (B, ) reward derived from measuring whether actions and gt_answer match
    '''
    rewards = []

    for action, gt_answer in zip(actions, gt_answers):
        gt_parsed = gt_answer.strip().lower()
        if action.strip().lower().split()[0] == gt_parsed or action.strip().lower().split()[-1] == gt_parsed:
            rewards.append(1)
        else:
            rewards.append(0)

    return torch.tensor(rewards)

# def retriever_reward_computing(actions, gt_paragraphs):

#     '''
#     Handles reward computing. 

#     Inputs:
#         * actions: B x words, LLM output
#         * gt_answer: B x words, ground truth output for LLM
    
#     Outputs:
#         * rewards: (B, ) reward derived from measuring whether actions and gt_answer match
#     '''
#     rewards = []

#     for action, gt_answer in zip(actions, gt_answers):
#         gt_parsed = gt_answer.strip().lower()
#         if action.strip().lower().split()[0] == gt_parsed or action.strip().lower().split()[-1] == gt_parsed:
#             rewards.append(1)
#         else:
#             rewards.append(0)

#     return torch.tensor(rewards)

def retriever_reward_computing(
        actions: torch.Tensor,          # (B, k) indices selected for each query
        gt_matrix: torch.Tensor,        # (B, M) graded relevance labels (0/1/2…)
        metric: str = "ndcg",           # "ndcg" | "precision" | "recall" | "sum"
) -> torch.Tensor:                      # → (B,) reward for each query
    """
    Compute retrieval rewards given selected columns and a ground-truth matrix.

    Parameters
    ----------
    actions   : LongTensor (B, k)
        Per-row indices of the passages picked by the retriever.
    gt_matrix : FloatTensor (B, M)
        Ground-truth relevance labels for every passage (same device).
    metric    : str
        Which reward to emit.  "ndcg" is the default and recommended.

    Returns
    -------
    rewards   : FloatTensor (B,)
        One reward per query (no grad).
    """
    B, k = actions.shape
    device = gt_matrix.device

    # -------- 1. Relevance scores of the chosen passages --------------------
    rels_sel = gt_matrix.gather(1, actions)          # (B, k) — graded 0/1/2 …

    # -------- 2. Different reward flavours ----------------------------------
    if metric == "ndcg":
        # DCG numerator
        denom = torch.log2(torch.arange(k, device=device) + 2).unsqueeze(0)  # (1, k)
        dcg  = (rels_sel / denom).sum(dim=1)                                 # (B,)

        # IDCG (ideal ranking) for normalisation
        sorted_rels, _ = torch.sort(gt_matrix, descending=True)
        ideal_topk = sorted_rels[:, :k] / denom
        idcg = ideal_topk.sum(dim=1)                                         # (B,)

        rewards = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))

    elif metric == "precision":
        rewards = (rels_sel > 0).float().sum(dim=1) / k

    elif metric == "recall":
        positives = (gt_matrix > 0).float().sum(dim=1).clamp(min=1)
        rewards = (rels_sel > 0).float().sum(dim=1) / positives

    else:  # plain sum of graded relevance
        rewards = rels_sel.sum(dim=1)

    return rewards.detach()              # reward is treated as a constant

def state_to_action_probs(queries, gt_paragraphs, retrieval_model):
    '''
    1. Deduplicate the facts.
    2. Encode the queries and deduplicated facts of the batch. Use facts from other queries as negative candidates.
    3. Retrive the passages with old policy. Sample actions, ask LLM for answer, and derive rewards based on the retrieval.

    Inputs:
        * queries: list of B queries
        * gt_paragraphs: list of B lists of dictionaries, each dictionary['text'] contains one fact
        * retrieval_model: the retrieval embedding model
    
    Outputs:
        * log_prob_sim_matrix: B x M, log softmax of sim_matrix
        * sim_matrix: B x M, each entry is the cosine similarity of the query and the passage
        * unique_passages: a list of B unique passages
        * E_q: B x D, each row is the retriever embedding of query
        * E_p_unique: M x D, each row is the retriever embedding of passage
    '''

    device = retrieval_model.device

    # 1. Load in a batch of queries and their corresponding facts. Deduplicate the facts.
    flat_passages    = [p['text'] for doc_list in gt_paragraphs for p in doc_list]

    unique_passages  = list(dict.fromkeys(flat_passages))  # preserves first‐seen order
    passage_to_idx   = {p: i for i, p in enumerate(unique_passages)}    # passage to index

    # 2. Encode the qeureis and deduplicated facts of the minibatch. Use facts from other queries as negative candidates.            
    queries_tokenized = retrieval_model.tokenize(queries)
    queries_tokenized = {k: v.to(device) for k, v in queries_tokenized.items()}
    E_q = retrieval_model(queries_tokenized)['sentence_embedding']        # (B x D)

    unique_passages_tokenized = retrieval_model.tokenize(unique_passages)
    unique_passages_tokenized = {k: v.to(device) for k, v in unique_passages_tokenized.items()}
    E_p_unique = retrieval_model(unique_passages_tokenized)['sentence_embedding'] 

    # 3. Retrive the top_k passages with old policy. Compute retrieval probabilities
    E_q_norm = torch.nn.functional.normalize(E_q,        p=2, dim=1)    # (B, D)
    E_p_norm = torch.nn.functional.normalize(E_p_unique, p=2, dim=1)    # (M, D)
    sim_matrix = E_q_norm @ E_p_norm.t()                                # (B, M)
    log_prob_sim_matrix = torch.log_softmax(sim_matrix, dim=1)  # shape (B, M)

    # Return shape: (B x M), (B x M), list of M passages, (B x D), (M x D)
    return log_prob_sim_matrix, sim_matrix, unique_passages, E_q, E_p_unique, passage_to_idx

def minibatch_indices(N: int, batch_size: int, *, generator: torch.Generator | None = None):
    """
    Yields 1-D index tensors of length ≤ batch_size that partition 0…N-1
    in random order (last chunk may be smaller).
    """
    perm = torch.randperm(N, generator=generator)       # shuffle once
    for start in range(0, N, batch_size):
        yield perm[start : start + batch_size]

def msmarco_collate_fn(batch):
    # batch is a list of tuples (query, paragraphs, scores)
    queries, paras, scores = zip(*batch)
    return list(queries), list(paras), list(scores)

class MSMARCO_dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        
        #### Download scifact.zip dataset and unzip the dataset
        dataset = "MSMARCO"
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        # out_dir = os.path.join('/data/richard/taggerv2/test/test6/beir/outputs', "datasets")
        data_path = '/data/richard/taggerv2/test/test6/beir/outputs/datasets/msmarco'

        try:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        except:
            raise ValueError(f'Wrong split {split}!')

        self.corpus = corpus

        self.corpus_keys = list(self.corpus.keys())
        self.corpus_values = list(self.corpus.values())
        self.queries = list(queries.values())
        self.qrels_values = list(qrels.values())

    def __getitem__(self, idx):

        query = self.queries[idx]
        doc2score = self.qrels_values[idx]            # a dict {doc_id: score}
        doc2score = {doc_id: int(score > 0) for doc_id, score in doc2score.items()}
        doc_ids, scores = zip(*doc2score.items())     # two tuples
        paragraphs     = [ self.corpus[d] for d in doc_ids ]
        return query, list(paragraphs), list(scores)

    def __len__(self):
        return len(self.queries)

# critic.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    """
    Inputs
    -------
    query_embed   : (B, D)  – query embeddings from the retriever
    corpus_embed  : (M, D)  – embeddings of *all* deduplicated passages
                              in the current batch corpus

    Output
    ------
    value_preds   : (B,)    – scalar V(s) for each query
    """
    def __init__(self, d_model: int):
        super().__init__()

        # Two-layer MLP on [q ; context] ∈ ℝ^{2D}
        self.value_head = nn.Sequential(
            nn.Linear(2 * d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, 1),
        )

    def forward(
        self,
        query_embed: torch.Tensor,   # (B, D)
        corpus_embed: torch.Tensor,  # (M, D)
    ) -> torch.Tensor:

        # detach ⇒ critic cannot back-prop into the retriever (actor)
        q = query_embed.detach()          # (B, D)
        c = corpus_embed.detach()         # (M, D)

        # ------------------------------------------------------------------
        # 1. Query–corpus attention  (soft summary, no top-K)
        # ------------------------------------------------------------------
        #   α_bm = softmax( q_b · c_m / √D )
        #   ctx_b = Σ_m α_bm · c_m
        # ------------------------------------------------------------------
        scale = 1.0 / math.sqrt(q.size(-1))
        attn_logits = torch.matmul(q, c.T) * scale        # (B, M)
        attn_weights = F.softmax(attn_logits, dim=-1)     # (B, M)
        context = torch.matmul(attn_weights, c)           # (B, D)

        # ------------------------------------------------------------------
        # 2. Value head on concatenated state embedding
        # ------------------------------------------------------------------
        state_repr = torch.cat([q, context], dim=-1)      # (B, 2D)
        values = self.value_head(state_repr).squeeze(-1)  # (B,)

        return values

