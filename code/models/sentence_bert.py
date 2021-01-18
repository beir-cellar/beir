from sentence_transformers import SentenceTransformer

class SentenceBERT:
    def __init__(self, model_name_or_path):
        self.model = SentenceTransformer(model_name_or_path)
    
    def encode_query(self, texts, show_progress_bar, batch_size, convert_to_tensor=False):
        return self.model.encode(
            sentences=texts, 
            show_progress_bar=show_progress_bar, 
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor)
    
    def encode_corpus(self, sentences, show_progress_bar, batch_size, convert_to_tensor=False):
        texts = [row["title"] + " - " + row["body"] for row in sentences]    
        return self.model.encode(
            sentences=texts, 
            show_progress_bar=show_progress_bar, 
            batch_size=batch_size, 
            convert_to_tensor=convert_to_tensor)