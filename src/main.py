from src.config import parse_args
from src.dataload.tweets_data_handler import TweetsDataHandler
from src.evaluation.evaluator import Evaluator
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
    if config['general']['model_checkpoint'] == '' or config['general']['trainer_checkpoint'] == '':
        # Start training from scratch
        model_builder = ModelBuilder(general_config=config['general'], model_config=config['model'])
        language_classifier = model_builder.build()
        trainer = Trainer(experiment_name=experiment_name)
        trainer.start_training(model=language_classifier,
                               tweets_data_handler=tweets_data_handler,
                               training_config=config['training'])
    else:
        # Resume training from scratch
        model_path = config['general']['model_checkpoint']
        trainer_path = config['general']['trainer_checkpoint']
        trainer = Trainer(experiment_name=experiment_name)
        trainer.resume_training(tweets_data_handler=tweets_data_handler,
                                training_config=config['training'],
                                model_path=model_path,
                                trainer_state_path=trainer_path)

elif config['general']['train_test_mode'] == 'test':
    model_path = config['general']['model_checkpoint']
    evaluator = Evaluator()
    evaluator.start_evaluation(tweets_data_handler=tweets_data_handler,
                               model_path=model_path,
                               training_config=config['training'])
else:
    raise ValueError('Invalid mode')
