from typing import List, Dict


import tqdm
import torch
import numpy as np
import transformers
from scipy import sparse


class SPLADE:
    def __init__(self, model_name_or_path, max_length=256):
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_length = max_length

    def encode(self, text):
        inputs = self.tokenizer(text, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs[0]
            attention_mask = inputs["attention_mask"]
            sentence_embedding = torch.max(torch.log(1 + torch.relu(token_embeddings)) * attention_mask.unsqueeze(-1), dim=1).values
        return sentence_embedding.cpu().numpy()

    def encode_query(self, query: str, **kwargs) -> sparse.csr_matrix:
        """ returns a csr_matrix of shape [1, n_vocab] """
        output = self.encode(query)
        return sparse.csr_matrix(output)
        
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> sparse.csr_matrix:
        """ returns a csr_matrix of shape [n_documents, n_vocab] """
        data, row, col = [], [], []
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        for i in tqdm.tqdm(range(0, len(sentences), batch_size), desc="encode_corpus"):
            batch = sentences[i:i+batch_size]
            dense = self.encode(batch)
            sparse_mat = sparse.coo_matrix(dense)
            data.extend(sparse_mat.data)
            row.extend(sparse_mat.row + i)
            col.extend(sparse_mat.col)
        shape = (max(row)+1, self.model.config.vocab_size)
        results = sparse.csr_matrix((data, (row, col)), shape=shape, dtype=np.float)
        return results
