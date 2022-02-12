from typing import List, Dict

import array
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
       
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, is_queries=False, **kwargs) -> sparse.csr_matrix:
        """ returns a csr_matrix of shape [n_documents, n_vocab] """
        # https://maciejkula.github.io/2015/02/22/incremental-construction-of-sparse-matrices/
        indices = array.array("i")
        indptr = array.array("i")
        data = array.array("f")
        sentences = [(doc["title"] + " " + doc["text"]).strip() for doc in corpus]
        indptr.append(0)
        last_indptr = 0
        for i in tqdm.tqdm(range(0, len(sentences), batch_size), desc="encode_corpus"):
            batch = sentences[i:i+batch_size]
            dense = self.encode(batch)
            nz_rows, nz_cols = np.nonzero(dense)
            nz_values = dense[(nz_rows, nz_cols)]
            data.extend(nz_values)
            local_indptr = np.bincount(nz_rows).cumsum() + last_indptr
            indptr.extend(local_indptr)
            indices.extend(nz_cols)
            last_indptr = local_indptr[-1]
        shape = (len(corpus), self.model.config.vocab_size)
        results = sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=np.float)
        return results
