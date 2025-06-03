import torch
import torch.nn.functional as F
import wandb
import timm
import argparse
import json
from tqdm import tqdm
from re_utils import MSMARCO_dataset, msmarco_collate_fn, llm_output, llm_reward_computing, minibatch_indices, state_to_action_probs, CriticNetwork, retriever_reward_computing
# from torchmetrics import MatthewsCorrCoef, F1Score
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import numpy as np
import random
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
parser.add_argument('--base_model', default='sentence-transformers/gtr-t5-xl', help='Base model type', type=str)
parser.add_argument('--llm_model_name', default='meta-llama/Llama-2-7b-chat-hf', help='Base model type', type=str)
parser.add_argument('--retrieve_top_k', default=3, help='Base model type', type=int)

# Data specs

# Training specs
parser.add_argument('--lr', default=1e-5, help='Standard learning rate for training', type=float)
parser.add_argument('--lr_scheduler', default=None, help='lr scheduler', type=str)
parser.add_argument('--epochs', default=30, help='Training epochs', type=int)
parser.add_argument('--wandb_project', default='RAG', help='Wandb project name', type=str)
parser.add_argument('--wandb_run_name', default=None, help='Wandb run name', type=str)
parser.add_argument('--batch_size', default=1024, help='Batch size for one GPU', type=int)
parser.add_argument('--minibatch_size', default=128, help='Batch size for one GPU', type=int)
parser.add_argument('--clip_eps', default=.2, help='Epsilon for clipped surrogate', type=float)
parser.add_argument('--iterations_per_batch', default=8, help='Batch size for one GPU', type=int)


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if __name__=='__main__':
    args = parser.parse_args()
    args.date_time = date_time
    print(f'Running args: {args}')

    # Load the dataset and create dataloader
    train_dataset = MSMARCO_dataset(split='train') 
    val_dataset = MSMARCO_dataset(split='test')
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2, collate_fn=msmarco_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2, collate_fn=msmarco_collate_fn)
    
    print('----------------- Data loaded -----------------')

    # Training statistics
    steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.num_processes)
    total_steps = steps_per_epoch * args.epochs 

    # Encoder loading
    retrieval_model = SentenceTransformer(args.base_model)      # gtr-t5-xl
    critic_model = CriticNetwork(d_model=1024)

    # # LLM loading
    # tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)
    # llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name).to(accelerator.device)

    # model ckpt loading
    if args.load_model_ckpt is not None:
        retrieval_model.load_state_dict(torch.load(args.load_model_ckpt))

    print('----------------- Model loaded -----------------')

    # Optimizer initialization
    retrieval_optimizer = torch.optim.Adam(retrieval_model.parameters(), lr=args.lr)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=args.lr)

    # Scheduler initialization
    if not args.lr_scheduler:
        retrieval_scheduler = torch.optim.lr_scheduler.ExponentialLR(retrieval_optimizer, gamma=1)
        critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=1)

    elif args.lr_scheduler == 'cycle':
        retrieval_scheduler = torch.optim.lr_scheduler.OneCycleLR(retrieval_optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
        critic_scheduler = torch.optim.lr_scheduler.OneCycleLR(critic_optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    elif args.lr_scheduler == 'cosine':
        retrieval_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(retrieval_optimizer, T_max=0.8*len(train_dataloader))
        critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=0.8*len(train_dataloader))

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
        num_trainable = sum([param.numel() for param in retrieval_model.parameters() if param.requires_grad]) + sum([param.numel() for param in critic_model.parameters() if param.requires_grad])
        print(f'num_trainable: {num_trainable}')

    # Prepare the variables for multi-gpu training
    # train_dataloader, val_dataloader, retrieval_model, critic_model, llm_model,  retrieval_optimizer, critic_optimizer, retrieval_scheduler, critic_scheduler = accelerator.prepare(
    #     train_dataloader, val_dataloader, retrieval_model, critic_model, llm_model, retrieval_optimizer, critic_optimizer, retrieval_scheduler, critic_scheduler
    # )
    train_dataloader, val_dataloader, retrieval_model, critic_model,  retrieval_optimizer, critic_optimizer, retrieval_scheduler, critic_scheduler = accelerator.prepare(
        train_dataloader, val_dataloader, retrieval_model, critic_model, retrieval_optimizer, critic_optimizer, retrieval_scheduler, critic_scheduler
    )

    # Start epochs
    retrieval_model.train()
    critic_model.train()
    # llm_model.eval()
    total_minibatch_idx = 0

    for epoch in range(args.epochs):

        '''
            for batch in dataloader:
                1. Load in a batch of queries and their corresponding facts. Deduplicate the facts.
                2. Encode the queries and deduplicated facts of the batch. Use facts from other queries as negative candidates.
                3. Retrive the passages with old policy. Sample actions, ask LLM for answer, and derive rewards based on the retrieval.
                4. Compute advantage based on reward - V(s).

                So far, the batch data contains:
                    * Queries (B x words)
                    * Decuplicated facts (M x words)
                    * Old encoded queries (B x D)
                    * Old unique encoded facts (M x D)
                    * Old actions (B x K, each entry is index of passage)
                    * Old action probabilities (B x 1) (Sum of log probabilities of retrieval)
                    * Old Rewards (B x 1)
                    * Advantage (B x 1)
                    * Critic-predicted values (B x 1)

                for minibatch in batch:
                    1. Sample a minibatch of queries and deduplicated facts. 
                    2. Retrive the passages with current policy. Compute probabilities of old actions based on current policy.
                    3. Compute action probability ratio with current and old action probabilities
                    4. Compute clipped surrogate loss with advantage, action probability ratio, and epsilon
                    5. Compute critic loss by MSE(Old rewards, V(s))
                    6. Update the current policy and critic with combined loss of clipped surrogate and critic loss

                5. After finishing all minibatches, we freeze the current policy as old policy for the next batch.    
                6. Set up eval for this batch
        '''

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Train', leave=True)

        for batch_idx, (queries, gt_paragraphs, scores) in pbar:
            
            with torch.no_grad():
                # (B x M), (B x M), list of M passages, (B x D), (M x D)
                log_prob_sim_matrix, sim_matrix, unique_passages, E_q, E_p_unique, passage_to_idx = state_to_action_probs(queries, gt_paragraphs, retrieval_model)

            # --- 3. Ground-truth matrix ----------------------------------------------
            gt_matrix = torch.zeros_like(sim_matrix)        # (B, M)   ←  all zeros by default

            for q_idx, (pass_list, score_list) in enumerate(zip(gt_paragraphs, scores)):
                # `pass_list`  : List[str] passages for this query
                # `score_list` : List[int]  relevance labels (0/1/2) aligned to pass_list
                for p, s in zip(pass_list, score_list):
                    col = passage_to_idx[p['text']]                 # lookup column in 0..M-1
                    gt_matrix[q_idx, col] = float(s)

            _, topk_idx = sim_matrix.topk(args.retrieve_top_k, dim=1, largest=True, sorted=True)  # (B, k)
            actions = topk_idx      # (B, k)

            selected_logp = log_prob_sim_matrix.gather(1, topk_idx)      # shape (B, k)
            sum_logp = selected_logp.sum(dim=1)            # shape (B,)
            action_probs = sum_logp.detach()        # in log of softmax prob
            
            # Sample actions and derive rewards based on the retrieval.
            query_facts = [
                [unique_passages[j] for j in row_idx]      # row_idx is a list of k indices
                for row_idx in topk_idx.tolist()
            ]       # a list of B lists of k passages

            # llm_answers = llm_output(llm_model, tokenizer, queries, query_facts)       # (B x words)
            rewards = retriever_reward_computing(actions, gt_matrix).to(accelerator.device)     # (B, )

            # queries: Queries (B x words)
            # unique_passages: Decuplicated facts (M x words)
            # E_q: Old encoded queries (B x D)
            # E_p_unique: Old encoded facts (M x D)
            # actions: Old actions (B x words)
            # action_probs: Old action probabilities (B x 1) (Sum / product of retrieval sigoid probabilities)
            # rewards: Old Rewards (B x 1)
            # values_pred: critic-predicted values (B x 1)

            real_batch_size = len(queries)
            minibatch_size = min(args.minibatch_size, real_batch_size//2)

            for idx in minibatch_indices(real_batch_size, minibatch_size):
                total_minibatch_idx += 1

                # 1. Sample a minibatch of queries and deduplicated facts. 
                mini_queries = [queries[int(index)] for index in idx]        # list of b questions
                # mini_gt_paragraphs = [gt_paragraphs[index] for index in idx]   # list of b paragraphs
                mini_gt_paragraphs = gt_paragraphs
                mini_actions = actions[idx]     # (b x k)
                mini_old_action_probs = action_probs[idx]   # (b, )
                mini_rewards = rewards[idx]         # (b, )

                # 2. Retrive the passages with current policy. Compute probabilities of old actions based on current policy.
                log_prob_sim_matrix, sim_matrix, unique_passages, E_q, E_p_unique, passage_to_idx = state_to_action_probs(mini_queries, mini_gt_paragraphs, retrieval_model)
                selected_logp = log_prob_sim_matrix.gather(1, mini_actions)      # shape (b, k)
                sum_logp = selected_logp.sum(dim=1)            # shape (b,)
                new_action_probs = sum_logp        # shape (b,)

                mini_values_pred = critic_model(E_q, E_p_unique)       # (b, ) Takes in the state: query and context available
                mini_advantages = mini_rewards - mini_values_pred     # (b, )

                # 3. Compute action probability ratio with current and old action probabilities
                prob_ratio = new_action_probs.exp().clamp_min(1e-12) / mini_old_action_probs.exp().clamp_min(1e-12)

                # 4. Compute clipped surrogate loss with advantage, action probability ratio, and epsilon                
                actor_loss = torch.min(prob_ratio * mini_advantages, torch.clamp(prob_ratio, 1 - args.clip_eps, 1 + args.clip_eps) * mini_advantages).mean()
            
                # 5. Compute critic loss by MSE(Old rewards, V(s)). 
                critic_loss = F.mse_loss(mini_values_pred, mini_rewards)

                # 6. Update the current policy and critic with combined loss of clipped surrogate and critic loss
                combined_loss = actor_loss + critic_loss
                retrieval_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                accelerator.backward(combined_loss)

                # Grad-norm check
                retrieval_grad_norm = torch.nn.utils.clip_grad_norm_(retrieval_model.parameters(), max_norm=1.0)
                retrieval_train_grad_norm = torch.mean(accelerator.gather(retrieval_grad_norm)).item()

                critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_norm=1.0)
                critic_train_grad_norm = torch.mean(accelerator.gather(critic_grad_norm)).item()

                current_lr = retrieval_optimizer.param_groups[0]['lr']

                retrieval_optimizer.step()
                critic_optimizer.step()

                retrieval_scheduler.step()
                critic_scheduler.step()

                if accelerator.is_main_process:
                    pbar.set_postfix(retrieval_loss=actor_loss, critic_loss=critic_loss)

                    wandb.log(
                        {
                            'train_lr': current_lr,
                            'train_actor_loss': actor_loss.item(),
                            'retrieval_train_grad_norm': retrieval_train_grad_norm,
                            'train_critic_loss': critic_loss.item(),
                        },
                        step=total_minibatch_idx,
                        commit=True
                    )

            if accelerator.is_main_process:
                    pbar.set_postfix(retrieval_loss=actor_loss, critic_loss=critic_loss)
                    wandb.log(
                        {
                            'train_reward': rewards.mean().item(),
                            'critic_train_grad_norm': critic_train_grad_norm,
                            'action_probs_average': action_probs.mean().item(),
                        },
                        step=total_minibatch_idx + 1,
                        commit=True
                    )
        # 5. After finishing all minibatches, we freeze the current policy as old policy for the next batch.  
        
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Val', leave=True)
        eval_rewards = []
        batch_critic_losses = []
        action_probs_list = []

        for batch_idx, (queries, gt_paragraphs, scores) in pbar:
            # (B x M), (B x M), list of M passages, (B x D), (M x D)
            log_prob_sim_matrix, sim_matrix, unique_passages, E_q, E_p_unique, passage_to_idx = state_to_action_probs(queries, gt_paragraphs, retrieval_model)

            _, topk_idx = sim_matrix.topk(args.retrieve_top_k, dim=1, largest=True, sorted=True)  # (B, k)
            actions = topk_idx

            selected_logp = log_prob_sim_matrix.gather(1, topk_idx)      # shape (B, k)
            sum_logp = selected_logp.sum(dim=1)            # shape (B,)
            action_probs = sum_logp
            
            # --- 3. Ground-truth matrix ----------------------------------------------
            gt_matrix = torch.zeros_like(sim_matrix)        # (B, M)   ←  all zeros by default

            for q_idx, (pass_list, score_list) in enumerate(zip(gt_paragraphs, scores)):
                # `pass_list`  : List[str] passages for this query
                # `score_list` : List[int]  relevance labels (0/1/2) aligned to pass_list
                for p, s in zip(pass_list, score_list):
                    col = passage_to_idx[p['text']]                 # lookup column in 0..M-1
                    gt_matrix[q_idx, col] = float(s)

            # Sample actions and derive rewards based on the retrieval.
            query_facts = [
                [unique_passages[j] for j in row_idx]      # row_idx is a list of k indices
                for row_idx in topk_idx.tolist()
            ]       # a list of B lists of k passages

            # llm_answers = llm_output(llm_model, tokenizer, queries, query_facts)        # (B x words)
            rewards = retriever_reward_computing(actions, gt_matrix).to(accelerator.device)     # (B, )

            # 4. Compute advantage based on reward - V(s).
            values_pred = critic_model(E_q, E_p_unique)       # (B, ) Takes in the state: query and context available
            advantage = rewards - values_pred
            batch_critic_loss = F.mse_loss(values_pred, rewards)

            eval_rewards.append(rewards.mean().item())
            batch_critic_losses.append(batch_critic_loss.item())
            action_probs_list.append(action_probs.mean().item())

        eval_reward_avg = sum(eval_rewards) / len(eval_rewards)
        batch_critic_avg = sum(batch_critic_losses) / len(batch_critic_losses)
        action_probs_avg = sum(action_probs_list) / len(action_probs_list)

        if accelerator.is_main_process:
            wandb.log(
                {
                    'eval_reward_avg': eval_reward_avg,
                    'critic_loss': batch_critic_avg,
                    'action_probs_average': action_probs_avg
                },
                step=total_minibatch_idx + 2,
                commit=True
            )










            















