import pandas as pd
import json

tsv_file = '/Users/kaengreg/Documents/Работа /НИВЦ/rus-mmarco/qrels/train.tsv'
df = pd.read_csv(tsv_file, sep='\t', nrows=100)

top100_tsv_file = '/Users/kaengreg/Documents/Работа /НИВЦ/rus-mmarco-mini/qrels/train.tsv'
df.to_csv(top100_tsv_file, sep='\t', index=False)

query_ids = df['query-id'].tolist()
corpus_ids = df['corpus-id'].tolist()

def load_jsonl(file_path, id_key):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())
            data[record[id_key]] = record
    return data

queries_file = '/Users/kaengreg/Documents/Работа /НИВЦ/rus-mmarco/queries.jsonl'
corpus_file = '/Users/kaengreg/Documents/Работа /НИВЦ/rus-mmarco/corpus.jsonl'

queries_data = load_jsonl(queries_file, '_id')
corpus_data = load_jsonl(corpus_file, '_id')

queries_output_file = '/Users/kaengreg/Documents/Работа /НИВЦ/rus-mmarco-mini/queries.jsonl'
corpus_output_file = '/Users/kaengreg/Documents/Работа /НИВЦ/rus-mmarco-mini/corpus.jsonl'

with open(queries_output_file, 'w', encoding='utf-8') as q_outfile, \
        open(corpus_output_file, 'w', encoding='utf-8') as c_outfile:

    for qid, cid in zip(query_ids, corpus_ids):
        query_record = queries_data.get(str(qid))
        corpus_record = corpus_data.get(str(cid))
        if query_record and corpus_record:
            q_outfile.write(json.dumps(query_record, ensure_ascii=False) + '\n')
            c_outfile.write(json.dumps(corpus_record, ensure_ascii=False) + '\n')

print(f'Successfully created {queries_output_file} and {corpus_output_file}')
