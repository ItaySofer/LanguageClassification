import torch
import torch.nn as nn
from flair.embeddings import CharacterEmbeddings, FlairEmbeddings, BertEmbeddings, StackedEmbeddings
from flair.data import Sentence


class MultilingualTextEmbedder(nn.Module):
    """ A block for extracting Natural Language Processing embeddings.
        The model may toggle on or off any concatenated combination of the following:
        - Character embeddings [Lample et al. 2016]
        - Flair contextual embeddings (forward or backward model) [Akbik et al. 2018]
        - BERT multilingual embeddings [Devlin et al. 2018]
    """

    def __init__(self,
                 use_character_embeddings=True,
                 use_flair_forward_embeddings=True,
                 use_flair_backward_embeddings=True,
                 use_bert_embeddings=False):
        """
        Creates a new instance of the embedder block.
        :param use_character_embeddings: Whether character embeddings should be extracted.
        :param use_flair_forward_embeddings: Whether Flair embeddings should be extracted (forward model).
        :param use_flair_backward_embeddings: Whether Flair embeddings should be extracted (backward model).
        :param use_bert_embeddings: Whether BERT embeddings should be extracted (multilingual model).
        """
        super().__init__()
        embeddings = []
        if use_character_embeddings:
            embeddings.append(CharacterEmbeddings())
        if use_flair_forward_embeddings:
            embeddings.append(FlairEmbeddings('multi-forward'))
        if use_flair_backward_embeddings:
            embeddings.append(FlairEmbeddings('multi-backward'))
        if use_bert_embeddings:
            embeddings.append(BertEmbeddings('bert-base-multilingual-cased'))

        self.embedder = StackedEmbeddings(embeddings)

    def forward(self, data):
        '''
        Embeds the data text with the configured NLP representations.
        :param data: Either:
        1) Input string, containing some semantic text.
        2) Iterable of strings, each containing some semantic text.
        Each entry in the data may be pre-tokenized or contain a whole phrase.
        When a whole phrase is given, it may get further tokenized by the embedder to produce embeddings.
        :return: A dict mapping token (string) -> single stacked representation (PyTorch tensor).
        Note: For pre-tokenized text, the embedder may break the tokens into even more sub-tokens before extracting
        embeddings (in which case, the dict will contain more than a single entry).
        In the case of multiple data entries, the returned type is a list of mappings.
        '''
        # Wrap as a sentence and embed it
        # Text may get further tokenized here..
        if isinstance(data, str):
            sentences = [Sentence(data)]                     # Single entry
        else:
            sentences = [Sentence(entry) for entry in data]  # List of entries

        self.embedder.embed(sentences)

        # For each entry: return embedding representation per token
        embeddings = []
        for sentence in sentences:
            token_embeddings = []
            for token in sentence:
                token_embeddings.append(token.embedding)
            embeddings.append(token_embeddings)
        return embeddings

    def embedding_length(self):
        return self.embedder.embedding_length
