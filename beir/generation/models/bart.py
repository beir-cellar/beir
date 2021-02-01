from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class BART:
    def __init__(self, model_path, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        self.eos_token_sep = ""
        self.gen_prefix = ""

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
    
    def generate(self, corpus, ques_per_passage):
        
        texts = [(self.gen_prefix + doc["title"] + " " + doc["text"] + self.eos_token_sep) for doc in corpus]
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
        return [self.tokenizer.decode(idx, skip_special_symbol=True) for idx in outs]