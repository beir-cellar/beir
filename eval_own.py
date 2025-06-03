import torch
import torch.nn.functional as F
import wandb
import timm
import argparse
import json
from tqdm import tqdm
from re_utils import MSMARCO_dataset, msmarco_collate_fn
from torch import linalg as LA

# from torchmetrics import MatthewsCorrCoef, F1Score
from sentence_transformers import SentenceTransformer

from accelerate import Accelerator
from datetime import datetime
import math
import os
import time

now = datetime.now()
date_time = now.strftime("%Y_%m_%d_%Hh%Mm%Ss")

accelerator = Accelerator()

parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument('--load_model_ckpt', default=None, help='Path to the model ckpt', type=str)
parser.add_argument('--base_model', default='sentence-transformers/gtr-t5-xl', help='Base model type', type=str)

# Data specs
parser.add_argument('--val_data_split', default=None, help='Split of the dataset', type=str)

# Training specs
parser.add_argument('--batch_size', default=64, help='Batch size for one GPU', type=int)
parser.add_argument('--retrieve_top_k', default=3, help='Batch size for one GPU', type=int)


if __name__=='__main__':
    args = parser.parse_args()
    args.date_time = date_time
    print(f'Running args: {args}')

    # Load the dataset and create dataloader
    val_dataset = MSMARCO_dataset(split=args.val_data_split)
        
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=2, collate_fn=msmarco_collate_fn)
    
    print('----------------- Data loaded -----------------')

    # Encoder loading
    model = SentenceTransformer(args.base_model, device=accelerator.device)      # gtr-t5-xl

    # model ckpt loading
    if args.load_model_ckpt is not None:
        model.load_state_dict(torch.load(args.load_model_ckpt))

    print('----------------- Model loaded -----------------')


    # Prepare the variables for multi-gpu training
    val_dataloader, model = accelerator.prepare(
        val_dataloader, model
    )

    # Start epochs
    model = accelerator.unwrap_model(model)
    model.train()


    model.eval()
    
    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Val', leave=True)

    if accelerator.is_main_process:
        val_loss_cur = []
        recall_ks = []
        sim_matrix_norms = []
        
    for batch_idx, (queries, paragraphs, scores) in pbar:
        with torch.no_grad():
            # Forward pass

            flat_passages    = [p['text'] for doc_list in paragraphs for p in doc_list]
            unique_passages  = list(dict.fromkeys(flat_passages))  # preserves first‐seen order
            passage_to_idx   = {p: i for i, p in enumerate(unique_passages)}    # passage to index

            # 3. encode all queries at once → (B x D)
            E_q        = model.encode(queries, convert_to_tensor=True)          # (B, D)
            E_p_unique = model.encode(unique_passages, convert_to_tensor=True)  # (M, D)

            # --- 2. Predicted sim-matrix (cosine) ------------------------------------
            E_q_norm = torch.nn.functional.normalize(E_q,        p=2, dim=1)    # (B, D)
            E_p_norm = torch.nn.functional.normalize(E_p_unique, p=2, dim=1)    # (M, D)
            sim_matrix = E_q_norm @ E_p_norm.t()                                # (B, M)
            sim_matrix_norm = LA.matrix_norm(sim_matrix).item()

            # --- 3. Ground-truth matrix ----------------------------------------------
            gt_matrix = torch.zeros_like(sim_matrix)        # (B, M)   ←  all zeros by default

            for q_idx, (pass_list, score_list) in enumerate(zip(paragraphs, scores)):
                # `pass_list`  : List[str] passages for this query
                # `score_list` : List[int]  relevance labels (0/1/2) aligned to pass_list
                for p, s in zip(pass_list, score_list):
                    col = passage_to_idx[p['text']]                 # lookup column in 0..M-1
                    gt_matrix[q_idx, col] = float(s)

            # boolean mask of ground-truth positives (label > 0)
            gt_pos_mask   = gt_matrix > 0                        # (B, M)
            num_pos_per_q = gt_pos_mask.sum(dim=1)               # (B,)

            # top-K passage indices per query under current similarity
            topk_idx = torch.topk(sim_matrix, args.retrieve_top_k, dim=1).indices  # (B, K)

            # how many of those K are relevant?
            hits = gt_pos_mask.gather(1, topk_idx).sum(dim=1)    # (B,)

            # avoid div-by-zero for queries with 0 positives
            denom = torch.clamp(num_pos_per_q, min=1)

            recall_per_q = hits.float() / denom                  # (B,)
            recall_k     = torch.mean(accelerator.gather(recall_per_q)).item()

            loss = torch.nn.functional.mse_loss(gt_matrix, sim_matrix) 

            val_loss = torch.mean(accelerator.gather(loss)).item()
            
        if accelerator.is_main_process:
            # Registering data for this eval
            val_loss_cur.append(val_loss)
            recall_ks.append(recall_k)
            sim_matrix_norms.append(sim_matrix_norm)

    if accelerator.is_main_process:
        # Register the valing metadata
        val_loss_avg = sum(val_loss_cur) / len(val_loss_cur)
        recall_k_avg = sum(recall_ks) / len(recall_ks)
        sim_matrix_norm_avg = sum(sim_matrix_norms) / len(sim_matrix_norms)

        print(f'val_loss_avg: {val_loss_avg}, recall_k_avg: {recall_k_avg}, sim_matrix_norm_avg:{sim_matrix_norm_avg}.')

    


    


















'''
accelerate launch \
--num_processes 1 \
--num_machines 1 \
--mixed_precision no \
--dynamo_backend no \
training.py \
--wandb_project RAG_run \
--lr 1e-5 \
--epochs 500 \
--lr_scheduler cycle \
--batch_size 256 \
'''







