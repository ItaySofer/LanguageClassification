from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import parse_args
from src.dataload.tweets_data_handler import TweetsDataHandler
import pickle
import os
import numpy as np
from typing import List, Union
from flair.data import Sentence
from flair.embeddings import TokenEmbeddings
import torch

config = parse_args()

tweets_data_handler = TweetsDataHandler(language_names=config['general']['langs'],
                                        training_config=config['training'])


class TfIdfEmbedder(TokenEmbeddings):
    def __init__(self):
        self.name = 'TfIdf'
        if os.path.exists('vectorizer.pickle'):
            self.char_vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
        else:
            self.char_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='char',
                ngram_range=(1, 4),
                dtype=np.float32,
                max_features=5000
            )
            train_dataloader = tweets_data_handler.create_train_dataloader(batch_size=np.inf, num_workers=0)
            b = next(iter(train_dataloader))

            self.char_vectorizer.fit_transform(b['tweet'])
            pickle.dump(self.char_vectorizer, open("vectorizer.pickle", "wb"))
        self.__embedding_length = len(self.char_vectorizer.idf_)
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def empty_embedding(self) -> np.ndarray:
        return np.zeros((3, 0, self.embedding_length))

    def batch_to_embeddings(self, lst):
        s = [self.char_vectorizer.transform(s).cuda(device=self.cuda_device) for s in lst]
        return s

    def embed_batch(self, batch: List[List[str]]) -> List[np.ndarray]:
        """
        Computes the TfIdf embeddings for a batch of tokenized sentences.
        Please note that ELMo has internal state and will give different results for the same input.
        See the comment under the class definition.
        Parameters
        ----------
        batch : ``List[List[str]]``, required
            A list of tokenized sentences.
        Returns
        -------
            A list of tensors, each representing the ELMo vectors for the input sentence at the same index.
        """
        embeddings = []

        # Batches with only an empty sentence will throw an exception inside AllenNLP, so we handle this case
        # and return an empty embedding instead.
        if batch == [[]]:
            embeddings.append(self.empty_embedding())
        else:
            embeds = self.batch_to_embeddings(batch)
            for i in range(len(batch)):
                length = 8
                # Slicing the embedding :0 throws an exception so we need to special case for empty sentences.
                if length == 0:
                    embeddings.append(self.empty_embedding())
                else:
                    embeddings.append(embeds[i, :, :length, :].detach().cpu().numpy())

        return embeddings

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        sentence_words: List[List[str]] = []
        for sentence in sentences:
            sentence_words.append([token.text for token in sentence])

        # embeddings = self.ee.embed_batch(sentence_words)
        embeddings = []
        for sentence in sentence_words:
            # sentence_words.append([token.text for token in sentence])
            embeddings.append(self.char_vectorizer.transform(sentence))

        for i, sentence in enumerate(sentences):

            sentence_embeddings = embeddings[i]

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token

                word_embedding = torch.FloatTensor(sentence_embeddings[token_idx].toarray()[0])

                token.set_embedding(self.name, word_embedding)

        return sentences

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:

        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        everything_embedded: bool = True

        self._add_embeddings_internal(sentences)

        return sentences
