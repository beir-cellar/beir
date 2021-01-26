import pathlib, os
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.util import download_url, unzip

dataset = "nfcorpus.zip"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}".format(dataset)

out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
os.makedirs(out_dir, exist_ok=True)

print("Downloading {} ...".format(dataset))
zip_file = os.path.join(out_dir, dataset)
download_url(url, zip_file)
unzip(zip_file, out_dir)

data_path = os.path.join(out_dir, dataset.replace(".zip", ""))
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

retriever = EvaluateRetrieval(model="dpr")
results = retriever.retrieve(corpus, queries, qrels)

ndcg, _map, recall = retriever.evaluate(qrels, results, retriever.k_values)
print(ndcg, _map, recall)