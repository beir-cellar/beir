# Majority of the code has been copied from PyGaggle MonoT5 implementation
# https://github.com/castorini/pygaggle/blob/master/pygaggle/rerank/transformer.py

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          PreTrainedModel,
                          PreTrainedTokenizer,
                          T5ForConditionalGeneration)
from typing import List, Union, Tuple, Mapping, Optional
from dataclasses import dataclass
from tqdm.autonotebook import trange
import torch


TokenizerReturnType = Mapping[str, Union[torch.Tensor, List[int],
                                         List[List[int]],
                                         List[List[str]]]]

@dataclass
class QueryDocumentBatch:
    query: str
    documents: List[str]
    output: Optional[TokenizerReturnType] = None

    def __len__(self):
        return len(self.documents)

class QueryDocumentBatchTokenizer:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 pattern: str = '{query} {document}',
                 **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.pattern = pattern
    
    def encode(self, strings: List[str]):
        assert self.tokenizer and self.tokenizer_kwargs is not None, \
                'mixin used improperly'
        ret = self.tokenizer.batch_encode_plus(strings,
                                               **self.tokenizer_kwargs)
        ret['tokens'] = list(map(self.tokenizer.tokenize, strings))
        return ret

    def traverse_query_document(
            self, batch_input: Tuple[str, List[str]], batch_size: int):
        query, doc_texts = batch_input[0], batch_input[1]
        for batch_idx in range(0, len(doc_texts), batch_size):
            docs = doc_texts[batch_idx:batch_idx + batch_size]
            outputs = self.encode([self.pattern.format(
                                        query=query,
                                        document=doc) for doc in docs])
            yield QueryDocumentBatch(query, docs, outputs)

class T5BatchTokenizer(QueryDocumentBatchTokenizer):
    def __init__(self, *args, **kwargs):
        kwargs['pattern'] = 'Query: {query} Document: {document} Relevant:'
        if 'return_attention_mask' not in kwargs:
            kwargs['return_attention_mask'] = True
        if 'padding' not in kwargs:
            kwargs['padding'] = 'longest'
        if 'truncation' not in kwargs:
            kwargs['truncation'] = True
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 512
        super().__init__(*args, **kwargs)


@torch.no_grad()
def greedy_decode(model: PreTrainedModel,
                  input_ids: torch.Tensor,
                  length: int,
                  attention_mask: torch.Tensor = None,
                  return_last_logits: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    decode_ids = torch.full((input_ids.size(0), 1),
                            model.config.decoder_start_token_id,
                            dtype=torch.long).to(input_ids.device)
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True)
        outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat([decode_ids,
                                next_token_logits.max(1)[1].unsqueeze(-1)],
                               dim=-1)
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids


class MonoT5:
    def __init__(self, 
                 model_path: str,
                 tokenizer: QueryDocumentBatchTokenizer = None,
                 use_amp = True,
                 token_false = None,
                 token_true  = None):
        self.model = self.get_model(model_path)
        self.tokenizer = tokenizer or self.get_tokenizer(model_path)
        self.token_false_id, self.token_true_id = self.get_prediction_tokens(
                model_path, self.tokenizer, token_false, token_true)
        self.model_path = model_path
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(model_path: str, *args, device: str = None, **kwargs) -> T5ForConditionalGeneration:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return AutoModelForSeq2SeqLM.from_pretrained(model_path, *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(model_path: str, *args, **kwargs) -> T5BatchTokenizer:
        return T5BatchTokenizer(
            AutoTokenizer.from_pretrained(model_path, use_fast=False, *args, **kwargs)
        )

    @staticmethod
    def get_prediction_tokens(model_path: str, tokenizer, token_false, token_true):
        if (token_false and token_true):
            token_false_id = tokenizer.tokenizer.get_vocab()[token_false]
            token_true_id  = tokenizer.tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id

    def predict(self, sentences: List[Tuple[str,str]], batch_size: int = 32, **kwargs) -> List[float]:
        
        sentence_dict, queries, scores = {}, [], []

        # T5 model requires a batch of single query and top-k documents
        for (query, doc_text) in sentences:
            if query not in sentence_dict:
                sentence_dict[query] = []
                queries.append(query) # Preserves order of queries
            sentence_dict[query].append(doc_text) 
        
        for start_idx in trange(0, len(queries), 1): # Take one query at a time
            batch_input = (queries[start_idx], sentence_dict[queries[start_idx]]) # (single query, top-k docs)            
            for batch in self.tokenizer.traverse_query_document(batch_input, batch_size): 
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    input_ids = batch.output['input_ids'].to(self.device)
                    attn_mask = batch.output['attention_mask'].to(self.device)
                    _, batch_scores = greedy_decode(self.model,
                                                    input_ids,
                                                    length=1,
                                                    attention_mask=attn_mask,
                                                    return_last_logits=True)

                    batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
                    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                    batch_log_probs = batch_scores[:, 1].tolist()
                    scores.extend(batch_log_probs)
        
        assert len(scores) == len(sentences) # Sanity check, should be equal
        return scores