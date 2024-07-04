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


file_path = "/Users/kaengreg/Documents/Работа /НИВЦ/rus-mmarco-mini/corpus.jsonl"
corpus = {}
with open(file_path, 'r') as file:
    for line in file:
        record = json.loads(line)
        record['processed_text'] = preprocess_text(record['text'])
        corpus[record['_id']] = record


es = Elasticsearch()
text_to_analyze = corpus['0']['processed_text']
analyze_response = es.indices.analyze(body={'text': text_to_analyze, 'analyzer': 'russian'})

print(json.dumps(analyze_response, indent=2, ensure_ascii=False))