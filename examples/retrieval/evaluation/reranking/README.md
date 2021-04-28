### Re-ranking BM25 top-100 using Cross-Encoder (Leaderboard)

In table below, we evaluate various different reranking architectures and evaluate them based on performance and speed. We include the following model architectures - 

- [MiniLM](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)
- [TinyBERT](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)


| Reranking-Model                        |Docs / Sec| MSMARCO | TREC-COVID | BIOASQ |NFCORPUS| NQ   |HOTPOT-QA| FIQA |SIGNAL-1M|
| ----------------------------------     |:------:  | :-----: | :--------: | :-----:|:------:| :--: |:------:| :--: |:--------:|
| **MiniLM Models**                      | |
| cross-encoder/ms-marco-MiniLM-L-2-v2   | 4100     |  0.373  |   0.669    | 0.471  | 0.337  |0.465 | 0.655  | 0.278| 0.334  |
| cross-encoder/ms-marco-MiniLM-L-4-v2   | 2500     |  0.392  |   0.720    | 0.516  | 0.358  |0.509 | 0.699  | 0.327| 0.350  |
| cross-encoder/ms-marco-MiniLM-L-6-v2   | 1800     |  0.401  |   0.722    | 0.529  | 0.360  |0.530 | 0.712  | 0.334| 0.351  |
| cross-encoder/ms-marco-MiniLM-L-12-v2  |  960     |  0.401  |   0.737    | 0.532  | 0.339  |0.531 | 0.717  | 0.336| 0.348  |
| **TinyBERT Models**                    | |
| cross-encoder/ms-marco-TinyBERT-L-2-v2 | 9000     |  0.354  |   0.689    | 0.466  | 0.346  |0.444 | 0.650  | 0.270| 0.338  |
| cross-encoder/ms-marco-TinyBERT-L-4    | 2900     |  0.371  |   0.640    | 0.470  | 0.323  |      | 0.679  | 0.260| 0.312  |
| cross-encoder/ms-marco-TinyBERT-L-6    | 680      |  0.380  |   0.652    | 0.473  | 0.339  |      | 0.682  | 0.305| 0.314  |
| cross-encoder/ms-marco-electra-base    | 340      |  0.384  |   0.667    | 0.489  | 0.303  |0.516 | 0.701  | 0.326| 0.308  |


| Reranking-Model                        |Docs / Sec| TREC-NEWS |ArguAna| Touche'20| DBPedia |SCIDOCS| FEVER |Clim.-FEVER| SciFact |
| -----------------------------------    |:-------: | :-------: |:-----:| :-----:  | :-----: |:-----:| :---: |:--------: | :-----: |
| **MiniLM Models**                      | |
| cross-encoder/ms-marco-MiniLM-L-2-v2   | 4100     |  0.417    | 0.157 |  0.363   |  0.502  | 0.145 | 0.759 |  0.215    |  0.607  |
| cross-encoder/ms-marco-MiniLM-L-4-v2   | 2500     |  0.431    | 0.430 |  0.371   |  0.531  | 0.156 | 0.775 |  0.228    |  0.680  |
| cross-encoder/ms-marco-MiniLM-L-6-v2   | 1800     |  0.436    | 0.415 |  0.349   |  0.542  | 0.164 | 0.802 |  0.240    |  0.682  |
| cross-encoder/ms-marco-MiniLM-L-12-v2  |  960     |  0.451    | 0.333 |  0.378   |  0.541  | 0.165 | 0.814 |  0.250    |  0.680  |
| **TinyBERT Models**                    | |
| cross-encoder/ms-marco-TinyBERT-L-2-v2 | 9000     |  0.385    | 0.341 |  0.311   |  0.497  | 0.151 | 0.647 |  0.173    |  0.662  |
| cross-encoder/ms-marco-TinyBERT-L-4    | 2900     |  0.377    | 0.398 |  0.333   |  0.350  | 0.149 | 0.760 |  0.194    |  0.658  |
| cross-encoder/ms-marco-TinyBERT-L-6    | 680      |  0.418    | 0.480 |  0.375   |  0.371  | 0.143 | 0.789 |  0.237    |  0.645  |
| cross-encoder/ms-marco-electra-base    | 340      |  0.430    | 0.313 |  0.378   |  0.380  | 0.154 | 0.793 |  0.246    |  0.524  |