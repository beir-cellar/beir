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


    service_name = "benchmark-ai-search"
    admin_key = os.environ["SEARCH_ADMIN_KEY"]

    index_name = dataset  + "-vector"

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
    # from azure.search.documents.models import (
    #     QueryAnswerType,
    #     QueryCaptionType,
    #     QueryLanguage,
    #     QueryType,
    #     RawVectorQuery,
    #     VectorizableTextQuery,
    #     VectorFilterMode,    
    # )
    from azure.search.documents.models import (
        QueryAnswerType,
        QueryCaptionType,
        QueryCaptionResult,
        QueryAnswerResult,
        SemanticErrorMode,
        SemanticErrorReason,
        SemanticSearchResultsType,
        QueryType,
        VectorizedQuery,
        VectorQuery,
        VectorFilterMode,    
    )
    from azure.search.documents.indexes.models import (  
        # AzureOpenAIEmbeddingSkill,  
        # AzureOpenAIParameters,  
        # AzureOpenAIVectorizer,  
        # ExhaustiveKnnParameters,  
        # ExhaustiveKnnVectorSearchAlgorithmConfiguration,
        ExhaustiveKnnAlgorithmConfiguration,
        ExhaustiveKnnParameters,
        FieldMapping,  
        # HnswParameters,  
        # HnswVectorSearchAlgorithmConfiguration,
        HnswAlgorithmConfiguration,
        HnswParameters,
        # IndexProjectionMode,  
        # InputFieldMappingEntry,  
        # OutputFieldMappingEntry,  
        # SemanticPrioritizedFieldsPrioritizedFields,    
        SearchField,  
        SearchFieldDataType,  
        SearchIndex,  
        SearchIndexer,  
        SearchIndexerDataContainer,  
        SearchIndexerDataSourceConnection,  
        # SearchIndexerIndexProjectionSelector,  
        # SearchIndexerIndexProjections,  
        # SearchIndexerIndexProjectionsParameters,  
        SearchIndexerSkillset,  
        SemanticConfiguration,  
        SemanticField,  
        # SemanticSettings,  
        SplitSkill,  
        VectorSearch,  
        VectorSearchAlgorithmKind,  
        VectorSearchAlgorithmMetric,  
        VectorSearchProfile,  
        SimpleField,
        SearchableField,
        CorsOptions,
        SemanticSearch,
        SemanticPrioritizedFields,
    ) 

    # Configure the vector search configuration  
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE
                )
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name="myExhaustiveKnn",
                kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                parameters=ExhaustiveKnnParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE
                )
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm_configuration_name="myExhaustiveKnn",
            )
        ]
    )

    fields = [
        SimpleField(name="corpusId", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SearchField(name="titleVector", type=SearchFieldDataType.Collection(
            SearchFieldDataType.Single), 
            vector_search_dimensions=1536, 
            vector_search_profile_name="myHnswProfile"), 
        SearchField(name="textVector", type=SearchFieldDataType.Collection(
            SearchFieldDataType.Single), 
            vector_search_dimensions=1536, 
            vector_search_profile_name="myHnswProfile"),  
    ]

    # import SemanticConfiguration from azure ai search
    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                # prioritized_keywords_fields=[SemanticField(field_name="Category")],
                content_fields=[SemanticField(field_name="text")]
        )
    )

    semantic_settings = SemanticSearch(configurations=[semantic_config])
    cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
    scoring_profiles = []
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_settings,
        scoring_profiles=scoring_profiles,
        cors_options=cors_options)

    print(index)

    try:
        result = admin_client.create_index(index)
        print ('Index', result.name, 'created')
    except Exception as ex:
        print ('Index creation error:', ex)

    # %%
    import openai
    from openai import AzureOpenAI
    from tenacity import retry, wait_random_exponential, stop_after_attempt  

    openai_client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = "2023-05-15",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    model = "text-embedding-ada-002-v2"

    # Generate Document Embeddings using OpenAI Ada 002
    # Read the text-sample.json
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    # Function to generate embeddings for title and content fields, also used for query embeddings
    def generate_embeddings(text, model=model):
        return openai_client.embeddings.create(input = [text], model=model).data[0].embedding

    # create documents for corpus
    documents = []
    for id in corpus:
        #print(id)
        documents.append({
            "corpusId": id,
            "title": corpus[id]["title"],
            "text": corpus[id]["text"],
            "titleVector": generate_embeddings(corpus[id]["title"]),
            "textVector": generate_embeddings(corpus[id]["text"])
        })
        break

    # %%
    try:
        # Upload documents to the index per 100 documents
        print("documents size is", len(documents))
        if len(documents) > 1000:
            for i in range(0, len(documents), 1000):
                result = search_client.upload_documents(documents=documents[i:i+1000])
                print("Upload of new document succeeded: {}".format(result[0].succeeded))
        else:
                result = search_client.upload_documents(documents=documents)
        #result = search_client.upload_documents(documents=documents)
        # print("Upload of new document succeeded: {}".format(result[0].succeeded))
    except Exception as ex:
        print (ex.message)


    # %% [markdown]
    # ## Search an index
    import time
    time.sleep(30)
    # %%
    results = search_client.search(search_text="*", include_total_count=True)
    print ('Total Documents Matching Query:', results.get_count())
    # for result in results:
    #     print(result)
    
    query = "Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging."
    vector_query = VectorizedQuery(
        vector=generate_embeddings(query),
        k_nearest_neighbors=3,
        fields="titleVector",
    )
    
    results = search_client.search(
        search_text=None, 
        vector_queries=[vector_query], 
        include_total_count=True, 
        top=5)
    for result in results:
        print(result["@search.score"])
    # %%

    # print("I like drug... with semantic search")
    # results = search_client.search(search_text="I like drug.", include_total_count=True, select='corpusId, title, text', top=5, semantic_configuration_name="my-semantic-config", query_type="semantic")
    # print('Total Documents Matching Query:', results.get_count())
    # for result in results:
    #     print(result.keys())
    #     print("@score:", result["@search.score"])
    #     print("@score.reranker_score", result["@search.reranker_score"])
    #     print("corpusId:", result["corpusId"])
    #     print("title", result["title"])
    #     print("text", result["text"][:100], "...\n")


    # print("I like drug... without semantic search")
    # results = search_client.search(search_text="I like drug.", include_total_count=True, select='corpusId, title, text', top=5)
    # print ('Total Documents Matching Query:', results.get_count())
    # for result in results:
    #     print("@score:", result["@search.score"])
    #     print("@score.reranker_score", result["@search.reranker_score"])
    #     print("corpusId:", result["corpusId"])
    #     print("title", result["title"])
    #     print("text", result["text"][:100], "...\n")

    # %%
    # query_ids = list(queries)
    # dict_results = {}
    # for query_id in query_ids:
    #     query = queries[query_id]
    #     results = search_client.search(search_text=query, include_total_count=True, select='corpusId, title, text', top=50)
    #     id_score = {}
    #     for result in results:
    #         id_score[result["corpusId"]] = result["@search.score"]
    #     # print(id_score)
    #     dict_results[query_id] = id_score

    # # %%
    # from beir.retrieval.evaluation import EvaluateRetrieval
    # ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, dict_results, [1, 3, 5, 10, 50, 100])
    # print(ndcg, _map, recall, precision)

    # import json
    # with open(dataset+"_azureaisearch_no_reranking_result.json", "w") as f:
    #     json.dump(dict_results, f)



    # query_ids = list(queries)
    # dict_results = {}
    # for query_id in query_ids:
    #     query = queries[query_id]
    #     results = search_client.search(search_text=query, include_total_count=True, select='corpusId, title, text', top=50, semantic_configuration_name="my-semantic-config", query_type="semantic")
    #     # results = search_client.search(search_text=query, include_total_count=True, select='corpusId, title, text', top=100, semantic_configuration_name="my-semantic-config", query_type="semantic")
    #     id_score = {}
    #     for result in results:
    #         # print("query:", query)
    #         # print(result["@search.reranker_score"])
    #         id_score[result["corpusId"]] = result["@search.reranker_score"]
    #     # print(id_score)
    #     dict_results[query_id] = id_score

    # # %%
    # from beir.retrieval.evaluation import EvaluateRetrieval
    # ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, dict_results, [1, 3, 5, 10, 50, 100])
    # print(ndcg, _map, recall, precision)
    # with open(dataset+"_azureaisearch_reranking_result.json", "w") as f:
    #     json.dump(dict_results, f)

# %%

def main():
    print("hello")
    load_dotenv()
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.CRITICAL,
                    handlers=[LoggingHandler()])
    # dataset_names = ["scifact", "nq", "scidocs", "arguana", "climate-fever", "dbpedia", "fever", "hotpotqa", "covid", "touche2020"]
    dataset_names = ["scifact"]
    # dataset_names = ["scifact", "scidocs", "arguana", "climate-fever"]

    for dataset_name in dataset_names:
        run(dataset_name)


if __name__ == "__main__":
    main()