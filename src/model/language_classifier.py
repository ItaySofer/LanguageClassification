import torch
import torch.nn as nn
from src.model.multilingual_text_embedder import MultilingualTextEmbedder
from src.model.mean_aggregator import MeanAggregator


class LanguageClassifier(nn.Module):
    def __init__(self, num_of_classes):
        super().__init__()
        self.embedder = MultilingualTextEmbedder()
        self.sequence_aggregator = MeanAggregator()

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
        token_embeddings = self.embedder(x)
        seq_embedding = self.sequence_aggregator(token_embeddings)
        y = self.projection(seq_embedding)

        return y
