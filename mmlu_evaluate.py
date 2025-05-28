from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Download MMLU dataset
mmlu_dataset = load_dataset("cais/mmlu", "all")

# Load MSMARCO corpus
msmarco_data_path = './beir/datasets/msmarco'
corpus, msmarco_queries, qrels = GenericDataLoader(data_folder=msmarco_data_path).load(split="dev")

# Get queries and answers from MMLU
mmlu_queries = {}
mmlu_choices = {}
mmlu_answers = {}
for i,data in enumerate(mmlu_dataset['dev']):
    mmlu_queries[i] = data['question']
    mmlu_choices[i] = data['choices']
    mmlu_answers[i] = data['choices'][data['answer']]

#### Load the SBERT model
# model = DRES(models.SentenceBERT("Alibaba-NLP/gte-modernbert-base"), batch_size=16)
model = DRES(models.SentenceBERT("BAAI/bge-large-en-v1.5"))
# model = DRES(models.SentenceBERT('sentence-transformers/gtr-t5-xl'))
# model = SentenceTransformer('sentence-transformers/gtr-t5-xl')      # gtr-t5-xl

# Retrieve documents using cosine-similarity for each query
retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "dot" for dot product
results = retriever.retrieve(corpus, mmlu_queries)

# Load LLM model
llm_model_name = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

# Evaluate answer performance
top_k = 5
correct = 0
total = 0

for i, question in tqdm(mmlu_queries.items(), desc="Evaluating Answers"):
    ground_truth_answer = mmlu_answers[i]
    choices = mmlu_choices[i]
    top_passages = sorted(results[i].items(), key=lambda x: x[1], reverse=True)[:top_k]
    passage_texts = [corpus[doc_id] for doc_id, _ in top_passages]
    context = "\n\n".join(passage_texts)
    
    prompt = f"Answer with just the text of the correct answer choice \n\n question: {question} \n choices:{choices} \n context: {context}"
    # prompt = f"Given the following context:\n\n{context}\n\n{question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_new_tokens=50)
    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if predicted_answer == ground_truth_answer:
        correct += 1
    else:
        print(f"Ground truth: {ground_truth_answer}")
        print(f"Predicted: {predicted_answer}")
    total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy}")