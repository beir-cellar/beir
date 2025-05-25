import torch
import wandb
import timm
import argparse
import json
from tqdm import tqdm
from re_utils import MSMARCO_dataset
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
parser.add_argument('--save_model_ckpt', default='/data/richard/taggerv2/test/test6/beir/outputs/ckpts', help='Path to save model ckpts', type=str)

# Data specs

# Training specs
parser.add_argument('--lr', default=1e-5, help='Standard learning rate for training', type=float)
parser.add_argument('--lr_scheduler', default=None, help='lr scheduler', type=str)
parser.add_argument('--epochs', default=30, help='Training epochs', type=int)
parser.add_argument('--wandb_project', default='RAG', help='Wandb project name', type=str)
parser.add_argument('--wandb_run_name', default=None, help='Wandb run name', type=str)
parser.add_argument('--batch_size', default=16, help='Batch size for one GPU', type=int)


if __name__=='__main__':
    args = parser.parse_args()
    args.date_time = date_time
    print(f'Running args: {args}')

    # Load the dataset and create dataloader
    train_dataset = MSMARCO_dataset(split='train') 
    val_dataset = MSMARCO_dataset(split='val')
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2)
    
    print('----------------- Data loaded -----------------')

    # Training statistics
    steps_per_epoch = math.ceil(len(train_dataloader) // accelerator.num_processes)
    total_steps = steps_per_epoch * args.epochs 

    # Encoder loading
    model = SentenceTransformer('sentence-transformers/gtr-t5-xl', device='cuda:0')      # gtr-t5-xl

    # model ckpt loading
    if args.load_model_ckpt is not None:
        model.load_state_dict(torch.load(args.load_model_ckpt))

    print('----------------- Model loaded -----------------')

    # Optimizer initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler initialization
    if not args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif args.lr_scheduler == 'cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0.8*len(train_dataloader))

    # Wandb initialization
    if accelerator.is_main_process:
        wandb_args = dict(
            entity="yrichard",
            project=args.wandb_project,
            name=(args.wandb_run_name + '_' + str(args.date_time)),
        )
        wandb.init(**wandb_args)
        # Log all hyperparameters from args
        wandb.config.update(vars(args))

        # Count the trainable parameters
        num_trainable = sum([param.numel() for param in model.parameters() if param.requires_grad])
        print(f'num_trainable: {num_trainable}')

    # Prepare the variables for multi-gpu training
    train_dataloader, val_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, val_dataloader, model, optimizer, scheduler
    )

    # Start epochs
    for epoch in range(args.epochs):

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Train', leave=True)
    
        # Training Loop
        for batch_idx, (queries, paragraphs, scores) in enumerate(train_dataloader):

            step_idx = batch_idx + epoch*steps_per_epoch

            # Forward pass
            # Encode the queries
            queries_emb = model.encode(queries) 

            flat_passages    = [p for doc_list in paragraphs for p in doc_list]
            unique_passages  = list(dict.fromkeys(flat_passages))  # preserves first‐seen order
            passage_to_idx   = {p: i for i, p in enumerate(unique_passages)}    # passage to index

            # 3. encode all queries at once → (B x D)
            E_q        = model.encode(queries, convert_to_tensor=True)          # (B, D)
            E_p_unique = model.encode(unique_passages, convert_to_tensor=True)  # (M, D)

            # --- 2. Predicted sim-matrix (cosine) ------------------------------------
            E_q_norm = torch.nn.functional.normalize(E_q,        p=2, dim=1)    # (B, D)
            E_p_norm = torch.nn.functional.normalize(E_p_unique, p=2, dim=1)    # (M, D)
            sim_matrix = E_q_norm @ E_p_norm.t()                                # (B, M)

            # --- 3. Ground-truth matrix ----------------------------------------------
            gt_matrix = torch.zeros_like(sim_matrix)        # (B, M)   ←  all zeros by default

            for q_idx, (pass_list, score_list) in enumerate(zip(paragraphs, scores)):
                # `pass_list`  : List[str] passages for this query
                # `score_list` : List[int]  relevance labels (0/1/2) aligned to pass_list
                for p, s in zip(pass_list, score_list):
                    col = passage_to_idx[p]                 # lookup column in 0..M-1
                    gt_matrix[q_idx, col] = float(s)

            loss = torch.nn.functional.mse_loss(gt_matrix, sim_matrix) 
            train_loss = torch.mean(accelerator.gather(loss)).item()

            # Compute grad
            accelerator.backward(loss)

            # Grad-norm check
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_grad_norm = torch.mean(accelerator.gather(grad_norm)).item()

            # Update
            optimizer.step()
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()

            if accelerator.is_main_process:
                pbar.set_postfix(loss=train_loss)

                wandb.log(
                    {
                        'train_lr': current_lr,
                        'train_loss': train_loss,
                        'train_grad_norm': train_grad_norm,
                    },
                    step=step_idx,
                    commit=True
                )

            torch.cuda.empty_cache()               # frees *cached* blocks so other code can reuse the memory
            # Validates the result on val set 10 times every epoch
            # print(step_idx, steps_per_epoch)
            if (step_idx + 1) % steps_per_epoch == 0:
            # if (batch_idx + 1) % 20 == 0:
                print(f'Start eval on epoch {epoch} batch index {batch_idx} ')
                model.eval()
                
                pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Val', leave=True)

                if accelerator.is_main_process:
                    val_loss_cur = []
                    
                for batch_idx, (queries, paragraphs, scores) in enumerate(train_dataloader):
                    with torch.no_grad():
                        # Forward pass
                        # Encode the queries
                        queries_emb = model.encode(queries) 

                        flat_passages    = [p for doc_list in paragraphs for p in doc_list]
                        unique_passages  = list(dict.fromkeys(flat_passages))  # preserves first‐seen order
                        passage_to_idx   = {p: i for i, p in enumerate(unique_passages)}    # passage to index

                        # 3. encode all queries at once → (B x D)
                        E_q        = model.encode(queries, convert_to_tensor=True)          # (B, D)
                        E_p_unique = model.encode(unique_passages, convert_to_tensor=True)  # (M, D)

                        # --- 2. Predicted sim-matrix (cosine) ------------------------------------
                        E_q_norm = torch.nn.functional.normalize(E_q,        p=2, dim=1)    # (B, D)
                        E_p_norm = torch.nn.functional.normalize(E_p_unique, p=2, dim=1)    # (M, D)
                        sim_matrix = E_q_norm @ E_p_norm.t()                                # (B, M)

                        # --- 3. Ground-truth matrix ----------------------------------------------
                        gt_matrix = torch.zeros_like(sim_matrix)        # (B, M)   ←  all zeros by default

                        for q_idx, (pass_list, score_list) in enumerate(zip(paragraphs, scores)):
                            # `pass_list`  : List[str] passages for this query
                            # `score_list` : List[int]  relevance labels (0/1/2) aligned to pass_list
                            for p, s in zip(pass_list, score_list):
                                col = passage_to_idx[p]                 # lookup column in 0..M-1
                                gt_matrix[q_idx, col] = float(s)

                        loss = torch.nn.functional.mse_loss(gt_matrix, sim_matrix) 

                        val_loss = torch.mean(accelerator.gather(loss)).item()
                       
                    if accelerator.is_main_process:
                        # Registering data for this eval
                        val_loss_cur.append(val_loss)

                if accelerator.is_main_process:
                    # Register the valing metadata
                    val_loss = sum(val_loss_cur) / len(val_loss_cur)
                    
                    wandb.log(
                        {
                            'val_loss': val_loss,
                        },
                        step=step_idx+2,
                        commit=True
                    )

                model.train() 

            if accelerator.is_main_process:
                if (batch_idx + 1) % (steps_per_epoch) == 0:
                    # Make a dir for ckpt saving 
                    ckpt_dir = os.path.join(args.save_model_ckpt, date_time)
                    os.makedirs(ckpt_dir, exist_ok=True)

                    print(f'Saving checkpoint on epoch {epoch} step index {step_idx}')
                    ckpt_path = os.path.join(ckpt_dir, f'model_step_{step_idx}.pth')
                    accelerator.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)


















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







