import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

class DPR:
    def __init__(self, model_name_or_path):
        
        # Query tokenizer and model
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.query_model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        
        # Context tokenizer and model
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.context_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    
    def encode_query(self, sentences, **kwargs):
        output = []
        batch_size = 16
        with torch.no_grad():
            for start_idx in tqdm.trange(0, len(texts), batch_size, desc='que'):
                encoded = self.query_tokenizer(texts[start_idx:start_idx+batch_size], truncation=True, padding=True, return_tensors='pt')
                model_out = self.query_model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                embeddings_q = model_out.pooler_output
                for emb in embeddings_q:
                    output.append(emb)

        out_tensor = torch.stack(output)
        assert out_tensor.shape[0] == len(sentences)
        return out_tensor
    
        
    def encode_corpus(self, sentences, **kwargs):
        output = []
        batch_size = 8
        with torch.no_grad():
            for start_idx in tqdm.trange(0, len(sentences), batch_size, desc='pas'):
                titles = [row['title'] for row in sentences[start_idx:start_idx+batch_size]]
                bodies = [row['body']  for row in sentences[start_idx:start_idx+batch_size]]
                encoded = self.tokenizer(titles, bodies, truncation='longest_first', padding=True, return_tensors='pt')
                model_out = self.model(encoded['input_ids'].cuda(), attention_mask=encoded['attention_mask'].cuda())
                embeddings_q = model_out.pooler_output.detach()
                for emb in embeddings_q:
                    output.append(emb)

        out_tensor = torch.stack(output)
        assert out_tensor.shape[0] == len(sentences)
        return out_tensor