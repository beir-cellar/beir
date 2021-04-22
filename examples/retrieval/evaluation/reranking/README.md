### Re-ranking using Cross-Encoder (Leaderboard)



| Model-Name                             |Docs / Sec| MSMARCO-Dev | TREC-COVID | BioASQ | NFCorpus | NQ   | HotpotQA | FIQA | Signal-1M |
| :----------------------------------    |:---------| :---------- | :--------- | :----- | :------- | :--  | :------- | :--- | :-------- |
| **Sentence Transformers Models**       |
| cross-encoder/ms-marco-electra-base    | 340      |  0.384      |   0.667    |  0.489 |   0.303  |0.516 | 0.701    | 0.326| 0.308     |
| cross-encoder/ms-marco-TinyBERT-L-2-v2 | 9000     |  0.354      |   0.689    |        |          |0.444 |          |      |           |
| cross-encoder/ms-marco-MiniLM-L-2-v2   | 4100     |  0.373      |   0.669    |        |          |0.465 |          |      |           |
| cross-encoder/ms-marco-MiniLM-L-4-v2   | 2500     |  0.392      |   0.720    |        |          |0.509 |          |      |           |
| cross-encoder/ms-marco-MiniLM-L-6-v2   | 1800     |  0.401      |   0.722    |        |          |0.530 |          |      |           |
| cross-encoder/ms-marco-MiniLM-L-12-v2  |  960     |             |   0.737    |        |          |0.531 |          |      |           |

| Model-Name                             |Docs / Sec| TREC-NEWS | ArguAna | Touche-2020| DBPedia | SCIDOCS | FEVER | Climate-FEVER | SciFact |
| :----------------------------------    |:---------| :-------- | :------ | :----------| :------ | :------ | :---- | :------------ | :------ |
| **Sentence Transformers Models**       |
| cross-encoder/ms-marco-electra-base    | 340      |  0.430    |  0.313  |  0.378     |  0.380  |  0.154  | 0.793 |  0.246        |  0.524  |
| cross-encoder/ms-marco-TinyBERT-L-2-v2 | 9000     |           |         |            |         |         |       |               |         |
| cross-encoder/ms-marco-MiniLM-L-2-v2   | 4100     |           |         |            |         |         |       |               |         |
| cross-encoder/ms-marco-MiniLM-L-4-v2   | 2500     |           |         |            |         |         |       |               |         |
| cross-encoder/ms-marco-MiniLM-L-6-v2   | 1800     |           |         |            |         |         |       |               |         |
| cross-encoder/ms-marco-MiniLM-L-12-v2  |  960     |           |         |            |         |         |       |               |         |