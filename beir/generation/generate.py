from tqdm.autonotebook import trange
import logging

logger = logging.getLogger(__name__)

class QueryGenerator:
    def __init__(self, model, **kwargs):
        self.model = model
        self.gen_q = {}

    # @staticmethod
    # def write_to_file(output_file, data):
    #     with open(output_file, 'w') as f:
    #         for corpus_id, questions in data.items():
    #             f.writelines('\t'.join([ques, corpus_id]) + '\n' for ques in list(questions))

    def generate(self, corpus, output_dir, ques_per_passage, prefix, batch_size=32):
        
        logger.info("Starting to Generate Questions...")
        logger.info("Batch Size: --- {} ---".format(batch_size))
        
        corpus_ids = list(corpus.keys())
        corpus = [corpus[doc_id] for doc_id in corpus_ids]

        # #output file configuration
        # filename = prefix + "qrels_sythetic.txt"
        # output_file = os.path.join(output_dir, filename)

        for start_idx in trange(0, len(corpus), batch_size, desc='pas'):            
            
            size = len(corpus[start_idx:start_idx + batch_size])
            
            queries = self.model.generate(
                corpus=corpus[start_idx:start_idx + batch_size], 
                ques_per_passage = ques_per_passage)
            
            assert len(queries) == size * ques_per_passage

            for idx in range(size):          
                
                # # Saving the generated questions (10000) at a time
                # if len(self.gen_q) % 1000 == 0:
                #     self.write_to_file(output_file, self.gen_q)

                corpus_id = corpus_ids[start_idx + idx]
                self.gen_q[corpus_id] = list()
                
                start_id = (start_idx + idx) * ques_per_passage
                end_id = start_id + ques_per_passage

                for query_generated in queries[start_id:end_id]:
                    try:
                        print(corpus_id, corpus[start_idx + idx], query_generated.strip())
                        self.gen_q[corpus_id].append(query_generated.strip())
                    except:
                        logger.error("error")
                        logger.error(batch_idx + idx, len(corpus_idx))  
        
        # # # Finally Saving all the generated questions
        # self.write_to_file(output_file, self.gen_q)