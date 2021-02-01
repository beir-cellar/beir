from sentence_transformers import SentenceTransformer, SentencesDataset, models, losses
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange
import logging
import time

logger = logging.getLogger(__name__)

class TrainRetriever:
    
    def __init__(self, model_name, model_save_path, batch_size=64, max_seq_length=350):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        self.eval = False

        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    def load_train(self, corpus, queries, qrels):
        query_ids = list(queries.keys())
        train_samples = []
        count = 0

        for start_idx in trange(0, len(query_ids), self.batch_size, desc='Adding Input Examples'):
            query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
            for query_id in query_ids_batch:
                for corpus_id, score in qrels[query_id].items():
                    if score >= 1: # if score = 0, we don't consider for training
                        s1 = queries[query_id]
                        s2 = corpus[corpus_id]
                        count += 1
                        train_samples.append(InputExample(guid=count, texts=[s1, s2], label=1))

        logger.info("Loaded {} training pairs.".format(len(train_samples)))
        return train_samples
    
    def load_eval(self, corpus, queries, qrels):
        pass


    def train(self, train_samples, eval_samples=None, num_epochs=1, lr=2e-5, evaluation_steps=5000):

        train_data = SentencesDataset(train_samples, model=self.model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)
        evaluator = SequentialEvaluator([], main_score_function=lambda x: time.time()) # dummy evaluator
        warmup_steps = int(len(train_samples) * num_epochs / self.batch_size * 0.1)

        if self.eval:
            evaluator = None
        
        # Train the model
        logger.info("Starting to train, Eval set present: {}...".format(self.eval))
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=num_epochs,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': lr},
                output_path=self.model_save_path,
                evaluation_steps=evaluation_steps,
                use_amp=True
                )