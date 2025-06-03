python train.py \
--wandb_run_name BGE_dev_bs3 \
--lr_scheduler cycle \
--base_model BAAI/bge-large-en-v1.5 \
--batch_size 128 \
--train_data_split dev \
--val_data_split dev \
--epochs 30