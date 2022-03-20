from typing import Optional, List, Dict, Union, Tuple
from transformers import BertConfig, BertModel, BertTokenizer, PreTrainedModel
import numpy as np
import torch
from tqdm.autonotebook import trange
from scipy.sparse import csr_matrix

class UniCOIL:
    def __init__(self, model_path: Union[str, Tuple] = None, sep: str = " ", query_max_length: int = 128, 
                doc_max_length: int = 500, **kwargs):
        self.sep = sep
        self.model = UniCoilEncoder.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert_input_emb = len(self.tokenizer.get_vocab())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length 
        self.model.to(self.device)
        self.model.eval()
    
    def encode_query(self, query: str, batch_size: int = 16, **kwargs):
        embedding = np.zeros(self.bert_input_emb, dtype=np.float)
        input_ids = self.tokenizer(query, max_length=self.query_max_length, padding='longest',
                                        truncation=True, add_special_tokens=True,
                                        return_tensors='pt').to(self.device)["input_ids"]
        
        with torch.no_grad():
            batch_weights = self.model(input_ids).cpu().detach().numpy()
            batch_token_ids = input_ids.cpu().detach().numpy()
            np.put(embedding, batch_token_ids, batch_weights.flatten())
        
        return embedding

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs):
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.encode(sentences, batch_size=batch_size, max_length=self.doc_max_length)
    
    def encode(
        self,
        sentences: Union[str, List[str], List[int]],
        batch_size: int = 32,
        max_length: int = 512) -> np.ndarray:

        passage_embs = []
        non_zero_tokens = 0
        
        for start_idx in trange(0, len(sentences), batch_size, desc="docs"):
            documents = sentences[start_idx: start_idx + batch_size]
            input_ids = self.tokenizer(documents, max_length=max_length, padding='longest',
                                        truncation=True, add_special_tokens=True,
                                        return_tensors='pt').to(self.device)["input_ids"]

            with torch.no_grad():
                batch_weights = self.model(input_ids).cpu().detach().numpy()
                batch_token_ids = input_ids.cpu().detach().numpy()
            
            for idx in range(len(batch_token_ids)):
                token_ids_and_embs = list(zip(batch_token_ids[idx], batch_weights[idx].flatten()))
                non_zero_tokens += len(token_ids_and_embs)
                passage_embs.append(token_ids_and_embs)
            
        col = np.zeros(non_zero_tokens, dtype=np.int)
        row = np.zeros(non_zero_tokens, dtype=np.int)
        values = np.zeros(non_zero_tokens, dtype=np.float)
        sparse_idx = 0    
        
        for pid, emb in enumerate(passage_embs):
            for tid, score in emb:
                col[sparse_idx] = pid
                row[sparse_idx] = tid
                values[sparse_idx] = score
                sparse_idx += 1

        return csr_matrix((values, (col, row)), shape=(len(sentences), self.bert_input_emb), dtype=np.float)

# class UniCOIL:
#     def __init__(self, model_path: Union[str, Tuple] = None, sep: str = " ", **kwargs):
#         self.sep = sep
#         self.model = UniCoilEncoder.from_pretrained(model_path)
#         self.tokenizer = BertTokenizer.from_pretrained(model_path)
#         self.sparse_vector_dim = len(self.tokenizer.get_vocab())
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model.to(self.device)
#         self.model.eval()

#     def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs):
#         max_length = 128  # hardcode for now
#         return self.encode(queries, batch_size=batch_size, max_length=max_length)

#     def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs):
#         max_length = 500
#         sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
#         return self.encode(sentences, batch_size=batch_size, max_length=max_length)
    
#     def encode(
#         self,
#         sentences: Union[str, List[str], List[int]],
#         batch_size: int = 32,
#         max_length: int = 512) -> np.ndarray:

#         embeddings = np.zeros((len(sentences), self.sparse_vector_dim), dtype=np.float)
        
#         for start_idx in trange(0, len(sentences), batch_size, desc="docs"):
#             documents = sentences[start_idx: start_idx + batch_size]
#             input_ids = self.tokenizer(documents, max_length=max_length, padding='longest',
#                                         truncation=True, add_special_tokens=True,
#                                         return_tensors='pt').to(self.device)["input_ids"]

#             with torch.no_grad():
#                 batch_weights = self.model(input_ids).cpu().detach().numpy()
#                 batch_token_ids = input_ids.cpu().detach().numpy()
            
#             for idx in range(len(batch_token_ids)):
#                 np.put(embeddings[start_idx + idx], batch_token_ids[idx], batch_weights[idx].flatten())

#         return embeddings
#         # return csr_matrix((values, (row, col)), shape=(len(sentences), self.sparse_vector_dim), dtype=np.float).toarray()


# Chunks of this code has been taken from: https://github.com/castorini/pyserini/blob/master/pyserini/encode/_unicoil.py
# For more details, please refer to uniCOIL by Jimmy Lin and Xueguang Ma (https://arxiv.org/abs/2106.14807)
class UniCoilEncoder(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = 'coil_encoder'
    load_tf_weights = None

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.tok_proj = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.bert.init_weights()
        self.tok_proj.apply(self._init_weights)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.bert.config.pad_token_id)
            )
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        tok_weights = self.tok_proj(sequence_output)
        tok_weights = torch.relu(tok_weights)
        return tok_weights