from transformers import T5Config, T5TokenizerFast, T5ForConditionalGeneration
import tensorflow as tf

import torch

class T5:
    def __init__(self, model_path, **kwargs):
        config = T5Config.from_pretrained('t5-base')
        self.tokenizer = T5TokenizerFast.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path, from_tf=True, config=config)
        
        self.eos_token_sep = ""
        self.gen_prefix = ""
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
    
    def generate(self, corpus, ques_per_passage):
        
        texts = [(self.gen_prefix + doc["title"] + " " + doc["text"] + self.eos_token_sep) for doc in corpus]
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            queries = self.model.generate(
                input_ids=encodings['input_ids'].to(self.device), 
                do_sample=True,
                max_length=64,
                top_k=25,
                top_p=0.95,
                num_return_sequences=ques_per_passage
                )
                
        return [self.tokenizer.decode(query, skip_special_symbol=True) for query in queries]