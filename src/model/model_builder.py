from src.model.multilingual_text_embedder import MultilingualTextEmbedder
from src.model.mean_aggregator import MeanAggregator
from src.model.language_classifier import LanguageClassifier


class ModelBuilder:

    def __init__(self, general_config, model_config):
        """
        Creates a new ModelBuilder instance, based on session configuration.
        :param general_config: General configuration of the session.
        :param model_config: Model configuration of the session.
        """
        self.general_config = general_config
        self.model_config = model_config

    def build(self):
        """
        Creates & initializes a LanguageClassifier model.
        :return: LanguageClassifier model, initialized and ready for training / evaluation.
        """
        embedder = self._build_embedder(self.model_config)
        aggregator = self._build_aggregator(self.model_config)

        num_of_classes = len(self.general_config['langs'])

        classifier = LanguageClassifier(num_of_classes=num_of_classes,
                                        embedder=embedder,
                                        aggregator=aggregator)

        return classifier

    @staticmethod
    def _build_embedder(model_config):
        """
        Builds an embedder block according to the model config
        :param model_config: A dict of model params, expected to contain a embedder argument
        :return: A MultilingualTextEmbedder block
        """
        embedder = MultilingualTextEmbedder(
            use_character_embeddings=('character' in model_config['embedder']),
            use_flair_forward_embeddings=('flair-fwd' in model_config['embedder']),
            use_flair_backward_embeddings=('flair-back' in model_config['embedder']),
            use_bert_embeddings=('bert' in model_config['embedder']),
            use_tfidf_embeddings=('tfidf' in model_config['embedder'])
        )

        return embedder

    @staticmethod
    def _build_aggregator(model_config):
        """
        Builds a sequence aggregator block according to the model config
        :param model_config: A dict of model params, expected to contain a aggregator argument
        :return: An aggregator block: nn.Module
        """
        possible_aggregators = {
            'mean': MeanAggregator
        }
        aggregator_name = model_config['aggregator']

        assert aggregator_name in possible_aggregators, \
            '%r is an unsupported model_aggregator type.' % aggregator_name

        aggregator = possible_aggregators[aggregator_name]()

        return aggregator

