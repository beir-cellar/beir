from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm.autonotebook import trange
import torch, logging, math, queue
import torch.multiprocessing as mp
from typing import List, Dict

logger = logging.getLogger(__name__)


class QGenModel:
    def __init__(self, model_path: str, gen_prefix: str = "", use_fast: bool = True, device: str = None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.gen_prefix = gen_prefix
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Use pytorch device: {}".format(self.device))
        self.model = self.model.to(self.device)
    
    def generate(self, corpus: List[Dict[str, str]], ques_per_passage: int, top_k: int, max_length: int, top_p: float = None, temperature: float = None) -> List[str]:
        
        texts = [(self.gen_prefix + doc["title"] + " " + doc["text"]) for doc in corpus]
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Top-p nucleus sampling
        # https://huggingface.co/blog/how-to-generate
        with torch.no_grad():
            if not temperature:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device), 
                    do_sample=True,
                    max_length=max_length,  # 64
                    top_k=top_k,  # 25
                    top_p=top_p,  # 0.95
                    num_return_sequences=ques_per_passage  # 1
                    )
            else:
                outs = self.model.generate(
                    input_ids=encodings['input_ids'].to(self.device), 
                    do_sample=True,
                    max_length=max_length,  # 64
                    top_k=top_k,  # 25
                    temperature=temperature,
                    num_return_sequences=ques_per_passage  # 1
                    )

        return self.tokenizer.batch_decode(outs, skip_special_tokens=True)
    
    def start_multi_process_pool(self, target_devices: List[str] = None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu']*4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(target=QGenModel._generate_multi_process_worker, args=(cuda_id, self.model, self.tokenizer, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}
    
    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()
    
    @staticmethod
    def _generate_multi_process_worker(target_device: str, model, tokenizer, input_queue, results_queue):
        """
        Internal working process to generate questions in multi-process setup
        """
        while True:
            try:
                id, batch_size, texts, ques_per_passage, top_p, top_k, max_length = input_queue.get()
                model = model.to(target_device)
                generated_texts = []
                
                for start_idx in trange(0, len(texts), batch_size, desc='{}'.format(target_device)):
                    texts_batch = texts[start_idx:start_idx + batch_size]
                    encodings = tokenizer(texts_batch, padding=True, truncation=True, return_tensors="pt")
                    with torch.no_grad():
                        outs = model.generate(
                            input_ids=encodings['input_ids'].to(target_device), 
                            do_sample=True,
                            max_length=max_length, # 64
                            top_k=top_k, # 25
                            top_p=top_p, # 0.95
                            num_return_sequences=ques_per_passage # 1
                            )
                    generated_texts += tokenizer.batch_decode(outs, skip_special_tokens=True)
                
                results_queue.put([id, generated_texts])
            except queue.Empty:
                break
    
    def generate_multi_process(self, corpus: List[Dict[str, str]], ques_per_passage: int, top_p: int, top_k: int, max_length: int, 
                               pool: Dict[str, object], batch_size: int = 32, chunk_size: int = None):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences
        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :return: Numpy matrix with all embeddings
        """

        texts = [(self.gen_prefix + doc["title"] + " " + doc["text"]) for doc in corpus]

        if chunk_size is None:
            chunk_size = min(math.ceil(len(texts) / len(pool["processes"]) / 10), 5000)

        logger.info("Chunk data into packages of size {}".format(chunk_size))

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for doc_text in texts:
            chunk.append(doc_text)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk, ques_per_passage, top_p, top_k, max_length])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk, ques_per_passage, top_p, top_k, max_length])
            last_chunk_id += 1

        output_queue = pool['output']
        
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])        
        queries = [result[1] for result in results_list]
        
        return [item for sublist in queries for item in sublist]