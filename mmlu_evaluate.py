from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm import tqdm

from collections import defaultdict
import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Download MMLU dataset
mmlu_dataset = load_dataset("cais/mmlu", "test")

# Load MSMARCO corpus
msmarco_data_path = './beir/datasets/msmarco'
corpus, msmarco_queries, qrels = GenericDataLoader(data_folder=msmarco_data_path).load(split="dev")

device = 'cuda:0'

# Get queries and answers from MMLU
mmlu_queries = {}
mmlu_choices = {}
mmlu_answers = {}
mmlu_subjects = {}

for i, data in enumerate(mmlu_dataset['test']):
    mmlu_queries[i] = data['question']
    mmlu_choices[i] = data['choices']
    mmlu_subjects[i] = data['subjects']
    mmlu_answers[i] = data['choices'][data['answer']]

#### Load the SBERT model
# model = DRES(models.SentenceBERT("Alibaba-NLP/gte-modernbert-base"), batch_size=16)
# model = DRES(models.SentenceBERT("BAAI/bge-large-en-v1.5"))
model = DRES(models.SentenceBERT('BAAI/llm-embedder'))
# model = DRES(models.SentenceBERT('sentence-transformers/gtr-t5-xl'))
# model = SentenceTransformer('sentence-transformers/gtr-t5-xl')      # gtr-t5-xl

# Retrieve documents using cosine-similarity for each query
retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "dot" for dot product
results = retriever.retrieve(corpus, mmlu_queries)

# Load LLM model
# llm_model_name = "google/flan-t5-xl"
llm_model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)

# Evaluate answer performance
top_k = 3
all_correct = 0
subject_correct = defaultdict(int)
subject_total = defaultdict(int)
total = 0

for i, question in tqdm(mmlu_queries.items(), desc="Evaluating Answers"):
    ground_truth_answer = mmlu_answers[i]
    choices = mmlu_choices[i]
    top_passages = sorted(results[i].items(), key=lambda x: x[1], reverse=True)[:top_k]
    passage_texts = [corpus[doc_id] for doc_id, _ in top_passages]
    context = "\n\n".join(passage_texts)
    subject = mmlu_subjects[i]
    
    prompt = f'''
        Knowledge:\n
        {context}\n

        The following is a multiple choice question about: {subject}\n
        {question}\n

        {choices}\n

    Answer with just the text of the all_correct answer choice. Answer:  
    
    '''
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**inputs, max_new_tokens=100)
    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    normalized_pred = predicted_answer.strip().lower().split()[0]  # take first token, lowercase
    normalized_true = ground_truth_answer.strip().lower()

    if normalized_pred == normalized_true:
        all_correct += 1
        subject_correct[subject] += 1
    else:
        # DEBUG
        print(f"Ground truth: {ground_truth_answer}")
        print(f"Predicted: {predicted_answer}")

    total += 1
    subject_total[subject] += 1

accuracy = all_correct / total
print(f"All Accuracy: {accuracy}")

for subject, correct in subject_correct.items():
    print(f'{subject} Accuracy: {correct / subject_total[subject]}')



