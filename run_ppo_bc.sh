python ppo_bc.py \
--wandb_run_name PPO_bc \
--lr_scheduler cycle \
--base_model BAAI/bge-large-en-v1.5 \
--batch_size 256 \
--minibatch_size 64 \
--epochs 10 \