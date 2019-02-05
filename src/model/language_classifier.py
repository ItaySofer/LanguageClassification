import torch
import torch.nn as nn


class LanguageClassifier(nn.Module):
    def __init__(self, num_of_classes, embedder, aggregator):
        super().__init__()
        self.embedder = embedder
        self.sequence_aggregator = aggregator

        embedding_length = self.embedder.embedding_length()

        self.projection = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_features=embedding_length, out_features=2048),
            nn.BatchNorm1d(num_features=2048),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=2048, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=512, out_features=num_of_classes)
        )

        # TODO: Handle config & architecture customization
        # TODO: Initialize weights
        # TODO: Embedding length should be controlled by the aggregator
        # TODO: Documentation

    def forward(self, x):
        """
        Classifies sentence string to it's respective language.
        :param x: An iterable of strings, each representing a sentence.
        :return: For each sentence - a label id of the language class identified by the classifier.
        """
        token_embeddings = self.embedder(x)
        seq_embedding = self.sequence_aggregator(token_embeddings)
        y = self.projection(seq_embedding)

        return y
