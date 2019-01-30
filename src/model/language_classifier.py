import torch
import torch.nn as nn
from src.model.multilingual_text_embedder import MultilingualTextEmbedder
from src.model.mean_aggregator import MeanAggregator


class LanguageClassifier(nn.Module):
    def __init__(self, num_of_classes, embedder, aggregator):
        super().__init__()
        self.embedder = embedder
        self.sequence_aggregator = aggregator

        embedding_length = self.embedder.embedding_length()

        self.projection = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_features=embedding_length, out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=1000, out_features=num_of_classes),
        )

        # TODO: Handle config & architecture customization
        # TODO: Initialize weights
        # TODO: Embedding length should be controlled by the aggregator
        # TODO: Documentation

    def forward(self, x):
        """
        Classifies a single token or list of tokens to their respective language.
        :param x: A single token (str) or an iterable of tokens.
        :return: Label id of the language class identified by the classifier.
        """

        # Handle both single strings as well as lists.
        # Let the embedder block break the phrase to tokens using whitespace as a delimeter.
        x = ' '.join(x) if not isinstance(x, str) else x

        token_embeddings = self.embedder(x)
        seq_embedding = self.sequence_aggregator(token_embeddings)
        y = self.projection(seq_embedding)

        return y
