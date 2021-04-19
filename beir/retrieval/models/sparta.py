from typing import List, Dict, Union, Tuple
from tqdm.autonotebook import trange
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import csr_matrix
import torch
import numpy as np

class SPARTA:
    def __init__(self, model_path: str = None, sep: str = " ", sparse_vector_dim: int = 2000, max_length: int = 500, **kwargs):
        self.sep = sep
        self.max_length = max_length
        self.sparse_vector_dim = sparse_vector_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.initialization()
        self.bert_input_embeddings = self._bert_input_embeddings()
    
    def initialization(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
    
    def _bert_input_embeddings(self):
        bert_input_embs = self.model.embeddings.word_embeddings(
            torch.tensor(list(range(0, len(self.tokenizer))), device=self.device))
        
        # Set Special tokens [CLS] [MASK] etc. to zero
        for special_id in self.tokenizer.all_special_ids:
            bert_input_embs[special_id] = 0 * bert_input_embs[special_id]
        
        return bert_input_embs
    
    def _compute_sparse_embeddings(self, documents):
        sparse_embeddings = []
        with torch.no_grad():
            tokens = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length).to(self.device)
            document_embs = self.model(**tokens).last_hidden_state
            for document_emb in document_embs:
                scores = torch.matmul(self.bert_input_embeddings, document_emb.transpose(0, 1))
                max_scores = torch.max(scores, dim=-1).values
                scores = torch.log(torch.relu(max_scores) + 1)
                top_results = torch.topk(scores, k=self.sparse_vector_dim)
                tids = top_results[1].cpu().detach().tolist()
                scores = top_results[0].cpu().detach().tolist()
                passage_emb = []
                
                for tid, score in zip(tids, scores):
                    if score > 0:
                        passage_emb.append((tid, score))
                    else:
                        break
                sparse_embeddings.append(passage_emb)

        return sparse_embeddings
    
    def encode_query(self, query: str, **kwargs):
        return self.tokenizer(query, add_special_tokens=False)['input_ids']
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 16, **kwargs):
        
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() for doc in corpus]
        sparse_idx = 0
        num_elements = len(sentences) * self.sparse_vector_dim
        col = np.zeros(num_elements, dtype=np.int)
        row = np.zeros(num_elements, dtype=np.int)
        values = np.zeros(num_elements, dtype=np.float)
        
        for start_idx in trange(0, len(sentences), batch_size, desc="docs"):
            doc_embs = self._compute_sparse_embeddings(sentences[start_idx: start_idx + batch_size])
            for doc_id, emb in enumerate(doc_embs):
                for tid, score in emb:
                    col[sparse_idx] = start_idx+doc_id
                    row[sparse_idx] = tid
                    values[sparse_idx] = score
                    sparse_idx += 1
                    
        return csr_matrix((values, (row, col)), shape=(len(self.bert_input_embeddings), len(sentences)), dtype=np.float)