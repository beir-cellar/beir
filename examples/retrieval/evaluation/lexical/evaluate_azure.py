from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from azure_search_documents import AzureSearchDocuments

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = "./datasets"
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")


service_name = "your-azure-search-service-name"
index_name = "your-index-name"
api_version = "2020-06-30"
api_key = "your-api-key"

azure_search = AzureSearchDocuments(service_name, index_name, api_version, api_key)

documents = []  # List of dictionaries representing your documents

for doc in corpus:
    document = {
        "id": doc['id'],
        "title": doc['title'],
        "text": doc['text']
    }
    documents.append(document)

azure_search.upload_documents(documents)

hostname = "your-azure-search-service-name.search.windows.net"
model = BM25(index_name=index_name, hostname=hostname)

retriever = EvaluateRetrieval(model)
retriever.retrieve(corpus, queries)


#%%
print("hello")
