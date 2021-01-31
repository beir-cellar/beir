from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from util import chunks
from typing import Dict

import tensorflow as tf
import torch

import tqdm
import random
import csv
import json
import argparse
import gzip
import io
import logging
import itertools
import pathlib, os
import pdb

class QueGenerator():
    def __init__(self, model_type, model_path):
        
        if model_type == "general":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.eos_token_sep = ""
            self.gen_prefix = ""

        elif model_type == "t5-tf":
            config = T5Config.from_pretrained('t5-base')
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_path, from_tf=True, config=config)
            self.eos_token_sep = ""
            self.gen_prefix = ""
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
    
    def _get_questions(self, texts, ques_per_passage):
        texts = [self.gen_prefix + text + self.eos_token_sep for text in texts]
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outs = self.model.generate(
                input_ids=encodings['input_ids'].to(self.device), 
                do_sample=True,
                max_length=64,
                top_k=25,
                top_p=0.95,
                num_return_sequences=ques_per_passage
                )
        dec = [self.tokenizer.decode(idx) for idx in outs]
        # pdb.set_trace()
        return dec

    def write_to_file(self,output_file, data):
        with open(output_file, 'w') as f:
            for corpus_id, questions in data.items():
                f.writelines('\t'.join([ques, corpus_id]) + '\n' for ques in list(questions))

    def generate(self,corpus: Dict[object, str], output_dir: str, ques_per_passage: int, batch_size: int, prefix: str):
        
        # print("Loading --- {} ---".format(model_type.upper()))
        print("Batch Size for Generation --- {} ---".format(batch_size))
        # query_generator_model = QueGenerator(model_type, model_path)
        # New questions generation
        generated_questions = {}
        logging.info("Starting to generate questions...")
        # corpus_idx, corpus_texts = [list(corpus.keys()), list(corpus.values())]
        corpus_idx = list(corpus.keys())[:10000]
        corpus_texts = [corpus[cid]["title"]+" "+corpus[cid]["text"] for cid in corpus_idx]

        batches = int(len(corpus_idx)/batch_size)
        total = batches if len(corpus_idx) % batch_size == 0 else batches + 1 
        batch_idx = 0

        #output file configuration
        filename = prefix + "qrels_sythetic.txt"
        output_file = os.path.join(output_dir, filename)

        for corpus_texts_chunk in tqdm.tqdm(chunks(corpus_texts, n=batch_size), total=total):
            
            size = len(corpus_texts_chunk)
            queries = self._get_questions(
                texts=corpus_texts_chunk,
                ques_per_passage = ques_per_passage)
            
            # pdb.set_trace()
            assert len(queries) == size * ques_per_passage

            for idx in range(size):            
                # Saving the generated questions (10000) at a time
                if len(generated_questions) % 1000 == 0:
                    self.write_to_file(output_file, generated_questions)

                corpus_id = corpus_idx[batch_idx + idx]
                # Changed it from set() to list()
                generated_questions[corpus_id] = list()
                start_idx = idx * ques_per_passage
                end_idx = start_idx + ques_per_passage

                for query_generated in queries[start_idx:end_idx]:
                    try:
                        generated_questions[corpus_id].append(query_generated.strip())
                    except:
                        print("error")
                        print(batch_idx + idx, len(corpus_idx))
            
            batch_idx += size
        
        logging.info("Training samples generated: {}".format(len(generated_questions)))    
        
        # # Finally Saving all the generated questions
        self.write_to_file(output_file, generated_questions)