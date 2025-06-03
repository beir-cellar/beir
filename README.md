## RAG-RL implementation

`evaluate.py`: Evalaution script for retrievers.

`train.py`: Training script for retrievers.

`ppo_bc.py`: PPO for behavior cloning.

`ppo.py`: PPO for end-to-end RL with LLM output.

`eval_own.py`: Recall-based retrieval evaluation on MSMARCO.

`re_utils.py`: Utility functions such as dataset.

`run_sft.sh`: Code to launch `train.py` for BC SFT training. 

`run_ppo_bc.sh`: Code to launch `ppo_bc.py` for BC PPO training. 

`run_eval_bc.sh`: Code to launch `eval_own.py` for BC evaluation.

Ignore the files containing `dev` -- they are my experiment files.

## Environment setup
```
git clone https://github.com/richard-guyunqi/RAG-RL.git
cd RAG-RL
pip install -e .
pip install sentence_transformers datasets pytrec_eval faiss-cpu
```

