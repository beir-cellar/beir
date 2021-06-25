from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from typing import Union, List, Dict, Tuple
from tqdm.autonotebook import trange
import torch

class DPR:
    def __init__(self, model_path: Union[str, Tuple] = None, **kwargs):
        # Query tokenizer and model
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_path[0])
        self.q_model = DPRQuestionEncoder.from_pretrained(model_path[0])
        self.q_model.cuda()
        self.q_model.eval()
        
        # Context tokenizer and model
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path[1])
        self.ctx_model = DPRContextEncoder.from_pretrained(model_path[1])
        self.ctx_model.cuda()
        self.ctx_model.eval()
    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> torch.Tensor:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                encoded = self.q_tokenizer(queries[start_idx:start_idx+batch_size], truncation=True, padding=True, return_tensors='pt')
                model_out = self.q_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                query_embeddings += model_out.pooler_output

        return torch.stack(query_embeddings)
        
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> torch.Tensor:
        
        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(corpus), batch_size):
                titles = [row['title'] for row in corpus[start_idx:start_idx+batch_size]]
                texts = [row['text']  for row in corpus[start_idx:start_idx+batch_size]]
                encoded = self.ctx_tokenizer(titles, texts, truncation='longest_first', padding=True, return_tensors='pt')
                model_out = self.ctx_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                corpus_embeddings += model_out.pooler_output.detach()
        
        return torch.stack(corpus_embeddings)