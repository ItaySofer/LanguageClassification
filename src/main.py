from src.config import parse_args
from src.dataload.tweets_data_handler import TweetsDataHandler
from src.model.model_builder import ModelBuilder
from src.training.trainer import Trainer
import os
import logging


def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger('language_classifier')
    logger.setLevel(log_level)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler = logging.StreamHandler()
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)


setup_logger()
config = parse_args()


experiment_name = config['general']['experiment_name']
tweets_data_handler = TweetsDataHandler(language_names=config['general']['langs'],
                                        data_root=config['training']['data_root'],
                                        train_test_split_perc=config['training']['train_test_split'])

if config['general']['train_test_mode'] == 'train':
    model_builder = ModelBuilder(general_config=config['general'], model_config=config['model'])
    language_classifier = model_builder.build()
    trainer = Trainer(experiment_name=experiment_name)
    trainer.start_training(model=language_classifier,
                           tweets_data_handler=tweets_data_handler,
                           training_config=config['training'])
elif config['general']['train_test_mode'] == 'test':
    pass
    # evaluator = Evaluator()
    # base_path = '../trained_models'
    # model_path = os.path.join(base_path, 'classifier_1_bkup.pt')
    # evaluator.start_evaluation(train_data, test_data, model_path)
else:
    raise ValueError('Invalid mode')
