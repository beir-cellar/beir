from transformers import BertLMHeadModel, BertTokenizer, DataCollatorWithPadding
from tqdm.autonotebook import trange
import torch, logging, math, queue
import torch.multiprocessing as mp
from typing import List, Dict
from nltk.corpus import stopwords
import numpy as np
import re

logger = logging.getLogger(__name__)

class TILDE:
    def __init__(self, model_path: str, gen_prefix: str = "", use_fast: bool = True, device: str = None, **kwargs):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=use_fast)
        self.model = BertLMHeadModel.from_pretrained(model_path)
        self.gen_prefix = gen_prefix
        _, self.bad_ids = self._clean_vocab(self.tokenizer)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Use pytorch device: {}".format(self.device))
        self.model = self.model.to(self.device)
    
    def _clean_vocab(self, tokenizer, do_stopwords=True):
        if do_stopwords:
            stop_words = set(stopwords.words('english'))
            # keep some common words in ms marco questions
            # stop_words.difference_update(["where", "how", "what", "when", "which", "why", "who"])
            stop_words.add("definition")

        vocab = tokenizer.get_vocab()
        tokens = vocab.keys()

        good_ids = []
        bad_ids = []

        for stop_word in stop_words:
            ids = tokenizer(stop_word, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                bad_ids.append(ids[0])

        for token in tokens:
            token_id = vocab[token]
            if token_id in bad_ids:
                continue

            if token[0] == '#' and len(token) > 1:
                good_ids.append(token_id)
            else:
                if not re.match("^[A-Za-z0-9_-]*$", token):
                    bad_ids.append(token_id)
                else:
                    good_ids.append(token_id)
        bad_ids.append(2015)  # add ##s to stopwords
        return good_ids, bad_ids
    
    def generate(self, corpus: List[Dict[str, str]], top_k: int, max_length: int) -> List[str]:
        
        expansions = []
        texts_batch = [(self.gen_prefix + doc["title"] + " " + doc["text"]) for doc in corpus]
        encode_texts = np.array(self.tokenizer.batch_encode_plus(
                texts_batch,
                max_length=max_length,
                truncation='only_first',
                return_attention_mask=False,
                padding='max_length')['input_ids'])
        
        encode_texts[:,0] = 1
        encoded_texts_gpu = torch.tensor(encode_texts).to(self.device)

        with torch.no_grad():
            logits = self.model(encoded_texts_gpu, return_dict=True).logits[:, 0]
            batch_selected = torch.topk(logits, top_k).indices.cpu().numpy()

            for idx, selected in enumerate(batch_selected):
                expand_term_ids = np.setdiff1d(np.setdiff1d(selected, encode_texts[idx], assume_unique=True), self.bad_ids, assume_unique=True)
                expansions.append(self.tokenizer.decode(expand_term_ids))
        
        return expansions