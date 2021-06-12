from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List, Dict

class QGenModel:
    def __init__(self, model_path: str, gen_prefix: str = "", use_fast: bool = True, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.gen_prefix = gen_prefix
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
    
    def generate(self, corpus: List[Dict[str, str]], ques_per_passage: int, top_p: int, top_k: int, max_length: int) -> List[str]:
        
        texts = [(self.gen_prefix + doc["title"] + " " + doc["text"]) for doc in corpus]
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Top-p nucleus sampling
        # https://huggingface.co/blog/how-to-generate
        with torch.no_grad():
            outs = self.model.generate(
                input_ids=encodings['input_ids'].to(self.device), 
                do_sample=True,
                max_length=max_length, # 64
                top_k=top_k, # 25
                top_p=top_p, # 0.95
                num_return_sequences=ques_per_passage # 1
                )
        
        return [self.tokenizer.decode(idx, skip_special_tokens=True) for idx in outs]