import torch
import torch.nn as nn


class MeanAggregator(nn.Module):
    """ A block for aggregating multiple token embeddings into a single sequence embedding.
        The scheme used for aggregation is "mean" over all token embeddings.
    """

    def __init__(self,):
        super().__init__()

    def forward(self, embeddings):
        """
        Averages all token embeddings in the sequence to produce a single sequence representation.
        :param embeddings: dict of token (string) -> embedding (PyTorch tensor)
        :return: A single PyTorch tensor representing the entire sequence
        """
        token_embeddings = [emb.unsqueeze(0) for emb in embeddings.values()]
        token_embeddings = torch.cat(token_embeddings, dim=0)
        if torch.cuda.is_available():
            token_embeddings = token_embeddings.cuda()

        mean_embedding = torch.mean(token_embeddings, dim=0)

        return mean_embedding
