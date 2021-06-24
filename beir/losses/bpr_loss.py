import math
import torch
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer, util

class BPRLoss(torch.nn.Module):
    """
        This loss expects as input a batch consisting of sentence triplets (a_1, p_1, n_1), (a_2, p_2, n_2)..., (a_n, p_n, n_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair. 
        You can also provide one or multiple hard negatives (n_1, n_2, ..) per anchor-positive pair by structering the data like this.
        
        We define the loss function as defined in ACL2021: Efficient Passage Retrieval with Hashing for Open-domain Question Answering.
        For more information: https://arxiv.org/abs/2106.00882
        
        Parts of the code has been reused from the source code of BPR (Binary Passage Retriever): https://github.com/studio-ousia/bpr.
        
        We combine two losses for training a binary code based retriever model =>
        1. Margin Ranking Loss: https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
        2. Cross Entropy Loss (or Multiple Negatives Ranking Loss): https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    """
    def __init__(self, model: SentenceTransformer, scale: float = 1.0, similarity_fct = util.dot_score, binary_ranking_loss_margin: float = 2.0, hashnet_gamma: float = 0.1):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, dot_score. Can also be set to cosine similarity.
        :param binary_ranking_loss_margin: margin used for binary loss. By default original authors found enhanced performance = 2.0, (Appendix D, https://arxiv.org/abs/2106.00882).
        :param hashnet_gamma: hashnet gamma function used for scaling tanh function. By default original authors found enhanced performance = 0.1, (Appendix B, https://arxiv.org/abs/2106.00882).
        """
        super(BPRLoss, self).__init__()
        self.global_step = 0
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.hashnet_gamma = hashnet_gamma
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.margin_ranking_loss = torch.nn.MarginRankingLoss(margin=binary_ranking_loss_margin)
    
    def convert_to_binary(self, input_repr: torch.Tensor) -> torch.Tensor:
        """
        The paper uses tanh function as an approximation for sign function, because of its incompatibility with backpropogation.
        """
        scale = math.pow((1.0 + self.global_step * self.hashnet_gamma), 0.5)
        return torch.tanh(input_repr * scale)

    def forward(self, sentence_features: Iterable[Dict[str, torch.Tensor]], labels: torch.Tensor):
        
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat([self.convert_to_binary(rep) for rep in reps[1:]])    
        
        # Dense Loss (or Multiple Negatives Ranking Loss)
        # Used to learn the encoder model
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        dense_loss = self.cross_entropy_loss(scores, labels)
        
        # Binary Loss (or Margin Ranking Loss)
        # Used to learn to binary coded model
        binary_query_repr = self.convert_to_binary(embeddings_a)
        binary_query_scores = torch.matmul(binary_query_repr, embeddings_b.transpose(0, 1))
        pos_mask = binary_query_scores.new_zeros(binary_query_scores.size(), dtype=torch.bool)
        for n, label in enumerate(labels):
            pos_mask[n, label] = True
        pos_bin_scores = torch.masked_select(binary_query_scores, pos_mask)
        pos_bin_scores = pos_bin_scores.repeat_interleave(embeddings_b.size(0) - 1)
        neg_bin_scores = torch.masked_select(binary_query_scores, torch.logical_not(pos_mask))
        bin_labels = pos_bin_scores.new_ones(pos_bin_scores.size(), dtype=torch.int64)
        binary_loss = self.margin_ranking_loss(
            pos_bin_scores, neg_bin_scores, bin_labels)
        
        self.global_step += 1
        
        return dense_loss + binary_loss
