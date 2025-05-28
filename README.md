## RAG-RL implementation

`evaluate.py`: Evalaution script for retrievers.

`train.py`: Training script for retrievers.

`re_utils.py`: Utility functions such as dataset.

`rush.sh`: Code to launch `train.py`.

Ignore the files containing `dev` -- they are my experiment files.

## Environment setup
```
git clone https://github.com/richard-guyunqi/RAG-RL.git
cd beir
pip install -e .
pip install sentence_transformers datasets pytrec_eval faiss-cpu
```

