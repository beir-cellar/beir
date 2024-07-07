import re
import pymorphy3
from nltk.corpus import stopwords
import nltk
import json
from elasticsearch import Elasticsearch

nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))
morph = pymorphy3.MorphAnalyzer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()

    processed_words = []
    for word in words:
        if word not in russian_stopwords:
            lemma = morph.parse(word)[0].normal_form
            processed_words.append(lemma)

    processed_text = ' '.join(processed_words)

    return processed_text


file_path = "/Users/kaengreg/Documents/Работа /НИВЦ/rus-mmarco-mini/queries.jsonl"
corpus = {}
with open(file_path, 'r') as file:
    for line in file:
        record = json.loads(line)
        record['processed_text'] = preprocess_text(record['text'])
        corpus[record['_id']] = record

with open(file_path, 'w', encoding='utf-8') as c_outfile:
    for cid in corpus.keys():
      c_outfile.write(json.dumps(corpus[cid], ensure_ascii=False) + '\n')


