# %%
import os
from dotenv import load_dotenv
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25


# %%
import logging


def run(dataset_name):
# %%
    print("dataset is ", dataset_name)
    dataset = dataset_name
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = "./datasets"
    data_path = util.download_and_unzip(url, out_dir)

    # %%
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")# pull data from corpus and queries

    # %%
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient 
    from azure.search.documents import SearchClient


    service_name = "beir-ai-search"
    admin_key = os.environ["SEARCH_ADMIN_KEY"]

    index_name = dataset

    # Create an SDK client
    endpoint = "https://{}.search.windows.net/".format(service_name)
    admin_client = SearchIndexClient(endpoint=endpoint,
                        index_name=index_name,
                        credential=AzureKeyCredential(admin_key))

    search_client = SearchClient(endpoint=endpoint,
                        index_name=index_name,
                        credential=AzureKeyCredential(admin_key))

    try:
        result = admin_client.delete_index(index_name)
        print ('Index', index_name, 'Deleted')
    except Exception as ex:
        print (ex)

    # %%
    from azure.search.documents.indexes.models import (
        CorsOptions,
        SearchableField,
        SimpleField,
        SearchIndex,
        SearchFieldDataType
    )


    fields = [
        SimpleField(name="corpusId", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="text", type=SearchFieldDataType.String),
    ]
    cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
    scoring_profiles = []
    index = SearchIndex(
        name=index_name,
        fields=fields,
        scoring_profiles=scoring_profiles,
        cors_options=cors_options)


    try:
        result = admin_client.create_index(index)
        print ('Index', result.name, 'created')
    except Exception as ex:
        print (ex)

    # %%
    # create documents for corpus
    documents = []
    for id in corpus:
        #print(id)
        documents.append({
            "corpusId": id,
            "title": corpus[id]["title"],
            "text": corpus[id]["text"]
        })

    # %%
    try:
        # Upload documents to the index per 100 documents
        print("documents size is", len(documents))
        if len(documents) > 1000:
            for i in range(0, len(documents), 1000):
                result = search_client.upload_documents(documents=documents[i:i+1000])
                print("Upload of new document succeeded (chunked): {}".format(result[0].succeeded))   
        else:
            result = search_client.upload_documents(documents=documents)
            print("Upload of new document succeeded: {}".format(result[0].succeeded))
        #result = search_client.upload_documents(documents=documents)
        # print("Upload of new document succeeded: {}".format(result[0].succeeded))
    except Exception as ex:
        print (ex.message)

    # %% [markdown]
    # ## Search an index

    # %%
    results = search_client.search(search_text="*", include_total_count=True)

    print ('Total Documents Matching Query:', results.get_count())
    # for result in results:
    #     print(result)

    # %%
    results = search_client.search(search_text="Micro", include_total_count=True, select='corpusId, title, text')

    print ('Total Documents Matching Query:', results.get_count())
    # for result in results:
    #     print("corpusId:", result["corpusId"])
    #     print("title", result["title"])
    #     print("text", result["text"][:100], "...\n")

    # %%
    query_ids = list(queries)
    dict_results = {}
    for query_id in query_ids:
        query = queries[query_id]
        results = search_client.search(search_text=query, include_total_count=True, select='corpusId, title, text', top=5)
        id_score = {}
        for result in results:
            id_score[result["corpusId"]] = result["@search.score"]
        # print(id_score)
        dict_results[query_id] = id_score

    # %%
    from beir.retrieval.evaluation import EvaluateRetrieval
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, dict_results, [1, 3, 5, 10, 50, 100, 1000])
    print(ndcg, _map, recall, precision)


# %%

def main():
    print("hello")
    load_dotenv()
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.CRITICAL,
                    handlers=[LoggingHandler()])
    dataset_names = ["scifact"]
    # dataset_names = ["scifact", "nq", "scidocs", "arguana", "climate-fever", "dbpedia", "fever", "hotpotqa", "covid", "touche2020"]
    for dataset_name in dataset_names:
        run(dataset_name)


if __name__ == "__main__":
    main()