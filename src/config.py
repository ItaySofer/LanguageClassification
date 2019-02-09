import argparse
import multiprocessing
from itertools import combinations, chain


def _restricted_float(x, min_val=0.0, max_val=1.0):
    """ Defines a type of restricted float with minimum / maximum values"""
    x = float(x)
    if x < min_val or x > max_val:
        raise argparse.ArgumentTypeError("%r not in range [%r, %r]"%(x, min_val, max_val))
    return x

def _allsubsets(choices):
    """ Returns a list of all subset lists of the choices list"""
    subsets = set()
    for combo_len in range(1, len(choices) + 1):
        subsets.update(chain(*[combinations(choices, ni) for ni in range(combo_len+1)]))
    subsets.remove(())
    return list(subsets)

def _parse_general_args(parser):
    group = parser.add_argument_group('general')
    group.add_argument('--train_test_mode',
                       choices=['train', 'test'],
                       default='train',
                       dest='general_train_test_mode',
                       help='Should the program run in train / test mode')
    group.add_argument('--experiment_name',
                       default='Default',
                       dest='general_experiment_name',
                       help='Unique identifier name for the experiment - used in plots & logs.')
    group.add_argument('--langs',
                       choices=['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl'],
                       default=['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl'],
                       nargs='+',
                       dest='general_langs',
                       help='Supported languages')
    group.add_argument('--model_checkpoint',
                       default='',
                       dest='general_model_checkpoint',
                       help='Path for *.pt of some saved model, used to resume training or for evaluation')
    group.add_argument('--trainer_checkpoint',
                       default='',
                       dest='general_trainer_checkpoint',
                       help='Path for *.pt of the trainer state, used to resume training')

    # For example, to resume training where you left off, use:
    # python main.py
    # --train_test_mode train
    # --model_checkpoint '../trained_models/last_model.pt'
    # --trainer_checkpoint '../trained_models/last_trainer_state.pt'

def _parse_model_args(parser):
    group = parser.add_argument_group('model')

    group.add_argument('--embedder',
                       default=['character', 'flair-fwd', 'flair-back'],
                       choices=['character', 'flair-fwd', 'flair-back', 'bert'],
                       nargs='+',
                       dest='model_embedder',
                       help='Combination of NLP embeddings to use in the embedder block')
    group.add_argument('--aggregator',
                       choices=['mean', 'bilstm', 'cnn'],
                       default='mean',
                       dest='model_aggregator',
                       help='Type of sequence aggregator to use for the model')


def _parse_training_args(parser):
    group = parser.add_argument_group('training')
    group.add_argument('--data_root',
                       default='../data',
                       type=str,
                       dest='training_data_root',
                       help='Path to root of data folder')
    group.add_argument('--trained_models_output_root',
                       default='../trained_models',
                       type=str,
                       dest='training_trained_models_output_root',
                       help='Path to root of where trained models are stored')
    group.add_argument('--train_test_split',
                       default=0.7,
                       type=_restricted_float,
                       dest='training_train_test_split',
                       help='Percentage of split between train / test data ~ [0.0, 1.0]')
    group.add_argument('--epochs',
                       default=50,
                       type=int,
                       dest='training_epochs',
                       help='Number of epochs to use during training')
    group.add_argument('--batch_size',
                       default=16,
                       type=int,
                       dest='training_batch_size',
                       help='Size of batch to use during training (# of tweet entries)')
    group.add_argument('--num_workers',
                       default=multiprocessing.cpu_count(),
                       type=int,
                       dest='training_num_workers',
                       help='Number of worker sub-processes to spawn for train / test dataloaders')
    group.add_argument('--criterion',
                       default='cross_entropy',
                       type=str,
                       choices=['cross_entropy'],
                       dest='training_criterion',
                       help='Loss function to use for training the classifier')
    group.add_argument('--optimizer',
                       default='adam',
                       type=str,
                       choices=['adam', 'sgd'],
                       dest='training_optimizer',
                       help='Optimizer to use for fancy back-propagation')
    group.add_argument('--learning_rate',
                       default=1e-3,
                       type=float,
                       dest='training_learning_rate',
                       help='Learning rate arg used b y the optimizer during training')
    group.add_argument('--log_rate',
                       default=200,
                       type=int,
                       dest='training_log_rate',
                       help='log_rate defines how many batches are processed between log dumps')
    group.add_argument('--train_data_metrics',
                       default=['avg_loss', 'accuracy',
                                'avg_precision', 'avg_recall', 'avg_f1', 'confusion_matrix'],
                       choices=[
                           'avg_loss', 'accuracy', 'max_batch_loss',
                           'avg_precision', 'avg_recall', 'avg_f1',
                           'precision_per_class', 'recall_per_class', 'f1_per_class',
                           'confusion_matrix'
                       ],
                       nargs='+',
                       dest='training_train_data_metrics',
                       help='Metrics to produce during training over training data')
    group.add_argument('--test_data_metrics',
                       default=[
                           'avg_loss', 'accuracy',
                           'avg_precision', 'avg_recall', 'avg_f1',
                           'precision_per_class', 'recall_per_class', 'f1_per_class',
                           'confusion_matrix'
                       ],
                       choices=[
                           'avg_loss', 'accuracy', 'max_batch_loss',
                           'avg_precision', 'avg_recall', 'avg_f1',
                           'precision_per_class', 'recall_per_class', 'f1_per_class',
                           'confusion_matrix'
                       ],
                       nargs='+',
                       dest='training_test_data_metrics',
                       help='Metrics to produce during validation over test data')
    group.add_argument('--tweet_transforms',
                       default=['remove_urls', 'remove_mentions', 'remove_response_token', 'remove_hashtags',
                                'remove_names', 'remove_non_alphachars'],
                       choices=['remove_urls', 'remove_mentions', 'remove_response_token', 'remove_hashtags',
                                'remove_names', 'remove_non_alphachars', 'to_lower_case'],
                       nargs='+',
                       dest='training_tweet_transforms',
                       help='Transforms to apply on tweets')
    group.add_argument('--pass_original_tweet',
                       default=False,
                       type=str2bool,
                       dest='training_pass_original_tweet',
                       help='indicates if Tweets Dataset should also pass the original tweet (for debugging)')


def _build_args_hierarchy(args):
    """
    Reconstruct the argparse results into a hierachy by looking at the args prefix.
    Ideally this should match the args groups.
    :param args: Namepsace of parsed args
    :return: Dict containing categories of (arg name -> arg val)
    """
    args_hierarchy = {}

    for arg_name, arg_val in vars(args).items():
        arg_category, arg_actual_name = arg_name.split('_', 1)

        if arg_category not in args_hierarchy:
            args_hierarchy[arg_category] = {}

        args_hierarchy[arg_category][arg_actual_name] = arg_val

    return args_hierarchy


def parse_args():
    """
    :return: Parses arguments from command line and constructs a hierarchical config dict
    """
    parser = argparse.ArgumentParser(description='A trainer / evaluator for language classifier based on tweets data.')
    _parse_general_args(parser)
    _parse_model_args(parser)
    _parse_training_args(parser)
    args = parser.parse_args()
    args = _build_args_hierarchy(args)
    return args


def str2bool(s):
    return bool(s)
