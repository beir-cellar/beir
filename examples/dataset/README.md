# Dataset Information

Generally, all public datasets can be easily downloaded using the zip folder.

Below we mention how to reproduce retrieval on datasets which are not public -

## 1. TREC-NEWS

### Corpus

1. Fill up the application to use the Washington Post (WaPo) Corpus: https://trec.nist.gov/data/wapost/
2. Loop through your contents. For a single document, get all the ``paragraph`` subtypes and extract HTML from text in case mime is ``text/html`` or directly include text from ``text/plain``.
3. I used ``html2text`` (https://pypi.org/project/html2text/) python package to extract text out of the HTML.

### Queries and Qrels
1. Download background linking topics and qrels from 2019 News Track: https://trec.nist.gov/data/news2019.html
2. We consider the document title as the query for our experiments.

## 2. BioASQ

### Corpus

1. Register yourself at BioASQ: http://www.bioasq.org/
2. Download documents from BioASQ task 9a (Training v.2020 ~ 14,913,939 docs) and extract the title and abstractText for each document.
3. There are few documents not present in this corpus but present in test qrels so we add them manually.
4. Find these manual documents here: https://docs.google.com/spreadsheets/d/1GZghfN5RT8h01XzIlejuwhBIGe8f-VaGf-yGaq11U-k/edit#gid=2015463710

### Queries and Qrels
1. Download Training and Test dataset from BioASQ 8B datasets which were published in 2020.
2. Consider all documents with answers as relevant (binary label) for a given question.

## 3. Robust04

### Corpus

1. Fill up the application to use the TREC disks 4 and 5: https://trec.nist.gov/data/cd45/index.html
2. Download, format it according to ``ir_datasets`` and get the preprocessed corpus: https://ir-datasets.com/trec-robust04.html#trec-robust04

### Queries and Qrels
1. Download the queries and qrels from ``ir_datasets`` with the key ``trec-robust04`` here - https://ir-datasets.com/trec-robust04.html#trec-robust04
2. For our experiments, we used the description of the query for retrieval.

## 4. Signal-1M

### Corpus
1. Scrape tweets from Twitter manually for the ids here: https://github.com/igorbrigadir/newsir16-data/tree/master/twitter/curated
2. I used ``tweepy`` (https://www.tweepy.org/) from python to scrape tweets. You can find the script here: [scrape_tweets.py](https://github.com/UKPLab/beir/blob/main/examples/dataset/scrape_tweets.py).
3. We preprocess the text retrieved, we remove emojis and links from the original text. You can find the function implementations in the code above.
4. Remove tweets which are empty or do not contain any text.

### Queries and Qrels
1. Sign up at Signal1M website to download qrels: https://research.signal-ai.com/datasets/signal1m-tweetir.html
2. Sign up at Signal1M website to download queries: https://research.signal-ai.com/datasets/signal1m.html
3. We consider the title of the query for our experiments.