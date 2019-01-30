import argparse


def _restricted_float(x, min_val=0.0, max_val=1.0):
    """ Defines a type of restricted float with minimum / maximum values"""
    x = float(x)
    if x < min_val or x > max_val:
        raise argparse.ArgumentTypeError("%r not in range [%r, %r]"%(x, min_val, max_val))
    return x


def _parse_general_args(parser):
    group = parser.add_argument_group('general')
    group.add_argument('--train_test_mode',
                       choices=['train', 'test'],
                       default='train',
                       dest='general_train_test_mode',
                       help='Should the program run in train / test mode')
    group.add_argument('--langs',
                       choices=['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl'],
                       default=['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl'],
                       dest='general_langs',
                       help='Supported languages')


def _parse_model_args(parser):
    group = parser.add_argument_group('model')

    group.add_argument('--embedder', nargs='+',
                       default=['character', 'flair-fwd', 'flair-back'],
                       choices=['character', 'flair-fwd', 'flair-back', 'bert'],
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
    group.add_argument('--train_test_split',
                       default=0.7,
                       type=_restricted_float,
                       dest='training_train_test_split',
                       help='Percentage of split between train / test data ~ [0.0, 1.0]')


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
