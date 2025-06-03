python eval_own.py \
--base_model BAAI/bge-large-en-v1.5 \
--val_data_split dev \
--batch_size 50000 \
--retrieve_top_k 1 \
--load_model_ckpt /data/richard/taggerv2/test/test6/beir/outputs/ckpts/2025_06_03_16h57m42s/model_step_2309.pth \
