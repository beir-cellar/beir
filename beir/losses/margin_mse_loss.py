from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor, nn


class MarginMSELoss(nn.Module):
    """
    Computes the Margin MSE loss between the query, positive passage and negative passage. This loss
    is used to train dense-models using cross-architecture knowledge distillation setup.

    Margin MSE Loss is defined as from (Eq.11) in Sebastian Hofstätter et al. in https://arxiv.org/abs/2010.02666:
    Loss(𝑄, 𝑃+, 𝑃−) = MSE(𝑀𝑠(𝑄, 𝑃+) − 𝑀𝑠(𝑄, 𝑃−), 𝑀𝑡(𝑄, 𝑃+) − 𝑀𝑡(𝑄, 𝑃−))
    where 𝑄: Query, 𝑃+: Relevant passage, 𝑃−: Non-relevant passage, 𝑀𝑠: Student model, 𝑀𝑡: Teacher model

    Remember: Pass the difference in scores of the passages as labels.
    """

    def __init__(self, model, scale: float = 1.0, similarity_fct="dot"):
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        scores_pos = (embeddings_query * embeddings_pos).sum(dim=-1) * self.scale
        scores_neg = (embeddings_query * embeddings_neg).sum(dim=-1) * self.scale
        margin_pred = scores_pos - scores_neg

        return self.loss_fct(margin_pred, labels)
