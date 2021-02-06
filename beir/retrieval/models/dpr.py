from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
import numpy as np
import torch
from typing import List, Dict
from tqdm.autonotebook import trange

class DPR:
    def __init__(self, q_model: str = None, ctx_model: str = None, **kwargs):
        # Query tokenizer and model
        self.query_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(q_model)
        self.query_model = DPRQuestionEncoder.from_pretrained(q_model)
        self.query_model.cuda()
        self.query_model.eval()
        
        # Context tokenizer and model
        self.context_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(ctx_model)
        self.context_model = DPRContextEncoder.from_pretrained(ctx_model)
        self.context_model.cuda()
        self.context_model.eval()
    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        output = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size, desc='que'):
                encoded = self.query_tokenizer(queries[start_idx:start_idx+batch_size], truncation=True, padding=True, return_tensors='pt')
                model_out = self.query_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                embeddings_q = model_out.pooler_output
                for emb in embeddings_q:
                    output.append(emb)

        out_tensor = torch.stack(output)
        assert out_tensor.shape[0] == len(queries)
        return np.asarray(out_tensor.cpu())
        
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        
        output = []
        with torch.no_grad():
            for start_idx in trange(0, len(corpus), batch_size, desc='pas'):
                titles = [row['title'] for row in corpus[start_idx:start_idx+batch_size]]
                texts = [row['text']  for row in corpus[start_idx:start_idx+batch_size]]
                encoded = self.context_tokenizer(titles, texts, truncation='longest_first', padding=True, return_tensors='pt')
                model_out = self.context_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                embeddings_c = model_out.pooler_output.detach()
                for emb in embeddings_c:
                    output.append(emb)

        out_tensor = torch.stack(output)
        assert out_tensor.shape[0] == len(corpus)
        return np.asarray(out_tensor.cpu())