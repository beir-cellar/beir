# BEIR Evaluation with ColBERT

In this example, we show how to evaluate the ColBERT zero-shot model on the BEIR Benchmark.

We modify the original [ColBERT](https://github.com/stanford-futuredata/ColBERT) repository to allow for evaluation of ColBERT across any BEIR dataset.

Please follow the required steps to evaluate ColBERT easily across any BEIR dataset.

## Installation with BEIR

- **Step 1**: Clone this beir-ColBERT repository (forked from original) which has modified for evaluating models on the BEIR benchmark: 
```bash
git clone https://github.com/NThakur20/beir-ColBERT.git
```

- **Step 2**: Create a new Conda virtual environment using the environment file provided: [conda_env.yml](https://github.com/NThakur20/beir-ColBERT/blob/master/conda_env.yml), It includes pip installation of the beir repository.
```bash
# https://github.com/NThakur20/beir-ColBERT#installation

conda env create -f conda_env.yml
conda activate colbert-v0.2
```
  - **Please Note**: We found some issues with ``_swigfaiss`` with both ``faiss-cpu`` and ``faiss-gpu`` installed on Ubuntu. If you face such issues please refer to: https://github.com/facebookresearch/faiss/issues/821#issuecomment-573531694 

## ``evaluate_beir.sh``

Run script ``evaluate_beir.sh`` for the complete evaluation of ColBERT model on any BEIR dataset. This scripts has five steps:

**1. BEIR Preprocessing**: We preprocess our BEIR data into ColBERT friendly data format using ``colbert/data_prep.py``. The script converts the original ``jsonl`` format to ``tsv``.  

```bash
python -m colbert.data_prep \
  --dataset ${dataset} \     # BEIR dataset you want to evaluate, for e.g. nfcorpus
  --split "test" \           # Split to evaluate on
  --collection $COLLECTION \ # Path to store collection tsv file
  --queries $QUERIES \       # Path to store queries tsv file
```

**2. ColBERT Indexing**: For fast retrieval, indexing precomputes the ColBERT representations of passages. 

**NOTE**: you will need to download the trained ColBERT model for inference

```bash
python -m torch.distributed.launch \
  --nproc_per_node=2 -m colbert.index \
  --root $OUTPUT_DIR \       # Directory to store the output logs and ranking files
  --doc_maxlen 300 \         # We work with 300 sequence length for document (unlike 180 set originally)
  --mask-punctuation \       # Mask the Punctuation
  --bsize 128 \              # Batch-size of 128 for encoding documents/tokens.
  --amp \                    # Using Automatic-Mixed Precision (AMP) fp32 -> fp16
  --checkpoint $CHECKPOINT \ # Path to the checkpoint to the trained ColBERT model 
  --index_root $INDEX_ROOT \ # Path of the root index to store document embeddings
  --index_name $INDEX_NAME \ # Name of index under which the document embeddings will be stored
  --collection $COLLECTION \ # Path of the stored collection tsv file
  --experiment ${dataset}    # Keep an experiment name
```
**3. FAISS IVFPQ Index**: We store and train the index using an IVFPQ faiss index for end-to-end retrieval. 

**NOTE**: You need to choose a different ``k`` number of partitions for IVFPQ for each dataset

```bash
python -m colbert.index_faiss \
  --index_root $INDEX_ROOT \     # Path of the root index where the faiss embedding will be store
  --index_name $INDEX_NAME \     # Name of index under which the faiss embeddings will be stored 
  --partitions $NUM_PARTITIONS \ # Number of Partitions for IVFPQ index (Seperate for each dataset (You need to chose)), for eg. 96 for NFCorpus 
  --sample 0.3 \                 # sample: 0.3
  --root $OUTPUT_DIR \           # Directory to store the output logs and ranking files
  --experiment ${dataset}        # Keep an experiment name
```

**4. Query Retrieval using ColBERT**: Retrieves top-_k_ documents, where depth = _k_ for each query.

**NOTE**: The output ``ranking.tsv`` file produced has integer document ids (because of faiss). Each each int corresponds to the doc_id position in the original collection tsv file. 

```bash
python -m colbert.retrieve \
  --amp \                        # Using Automatic-Mixed Precision (AMP) fp32 -> fp16
  --doc_maxlen 300 \             # We work with 300 sequence length for document (unlike 180 set originally)
  --mask-punctuation \           # Mask the Punctuation
  --bsize 256 \                  # 256 batch-size for evaluation
  --queries $QUERIES \           # Path which contains the store queries tsv file
  --nprobe 32 \                  # 32 query tokens are considered
  --partitions $NUM_PARTITIONS \ # Number of Partitions for IVFPQ index
  --faiss_depth 100 \            # faiss_depth of 100 is used for evaluation (Roughly 100 top-k nearest neighbours are used for retrieval)
  --depth 100 \                  # Depth is kept at 100 to keep 100 documents per query in ranking file
  --index_root $INDEX_ROOT \     # Path of the root index of the stored IVFPQ index of the faiss embeddings
  --index_name $INDEX_NAME \     # Name of index under which the faiss embeddings will be stored 
  --checkpoint $CHECKPOINT \     # Path to the checkpoint to the trained ColBERT model 
  --root $OUTPUT_DIR \           # Directory to store the output logs and ranking files
  --experiment ${dataset} \      # Keep an experiment name
  --ranking_dir $RANKING_DIR     # Ranking Directory will store the final ranking results as ranking.tsv file
```

**5. Evaluation using BEIR**: Evaluate the ``ranking.tsv`` file using the BEIR evaluation script for any dataset.

```bash
python -m colbert.beir_eval \
  --dataset ${dataset} \                   # BEIR dataset you want to evaluate, for e.g. nfcorpus
  --split "test" \                         # Split to evaluate on
  --collection $COLLECTION \               # Path of the stored collection tsv file 
  --rankings "${RANKING_DIR}/ranking.tsv"  # Path to store the final ranking tsv file 
```
