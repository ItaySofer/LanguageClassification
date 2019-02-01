import torch
import torch.nn as nn


class MeanAggregator(nn.Module):
    """ A block for aggregating multiple token embeddings into a single sequence embedding.
        The scheme used for aggregation is "mean" over all token embeddings.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _infer_embedding_size(embeddings):
        return list(embeddings[0].values())[0].shape[0]

    def forward(self, embeddings):
        """
        Averages all token embeddings in the sequence to produce a single sequence representation.
        :param embeddings: dict of token (string) -> embedding (PyTorch tensor)
        :return: A single PyTorch tensor representing the entire sequence
        """

        embeddings_size = self._infer_embedding_size(embeddings)
        mean_embeddings = torch.zeros(len(embeddings), embeddings_size)

        for sentence_idx, single_entry_embeddings in enumerate(embeddings):
            token_embeddings = [emb.unsqueeze(0) for emb in single_entry_embeddings.values()]
            token_embeddings = torch.cat(token_embeddings, dim=0)

            mean_embedding = torch.mean(token_embeddings, dim=0)
            mean_embeddings[sentence_idx] = mean_embedding

        if torch.cuda.is_available():
            mean_embeddings = mean_embeddings.cuda()

        return mean_embeddings
