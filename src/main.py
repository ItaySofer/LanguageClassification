from src.config import parse_args
from torchvision import transforms
from src.dataload.tweets_dataset import TweetsDataset
from src.dataload.tweet_transforms import *
from src.dataload.label_transforms import ToLabelIdTensor
from src.model.model_builder import ModelBuilder
from src.training.trainer import Trainer
import torch
import torch.nn as nn
import logging
from torch.utils.data import ConcatDataset, Subset, DataLoader

def setup_logger():
    logger = logging.getLogger('language_classifier')
    handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def prepare_data(data_root, language_names, tweet_transform, label_transform, train_percent):
    train_datasets = []
    test_datasets = []
    for language_name in language_names:
        dataset = TweetsDataset(root_dir=data_root, language_name=language_name,
                                tweet_transform=tweet_transform, label_transform=label_transform)

        train_size = int(train_percent * len(dataset))
        train_dataset = Subset(dataset, range(0, train_size))
        test_dataset = Subset(dataset, range(train_size, len(dataset)))

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    return ConcatDataset(train_datasets), ConcatDataset(test_datasets)


setup_logger()
config = parse_args()

language_names = config['general']['langs']
train_test_split_perc = config['training']['train_test_split']
data_root = config['training']['data_root']

tweet_transform = transforms.Compose(
    [SplitToWords(),
     RemoveURLs(),
     RemoveMentions(),
     RemoveResponseToken(),
     RemoveHashtags(),
     CleanTokens(),
     RemoveBlanks(),
     ToLowerCase(),
     JoinWordsToSentence()])
label_transform = ToLabelIdTensor(language_names=language_names)
train_data, test_data = prepare_data(data_root=data_root,
                                     language_names=language_names,
                                     tweet_transform=tweet_transform,
                                     label_transform=label_transform,
                                     train_percent=train_test_split_perc)


train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=16)
# test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
#
# dataloaders = [train_dataloader, test_dataloader]
# for dataloader in dataloaders:
#     for i, sample_batched in enumerate(dataloader):
#         print(sample_batched)
#         if i == 2:
#             break

model_builder = ModelBuilder(general_config=config['general'], model_config=config['model'])
classifier = model_builder.build()

trainer = Trainer()
trainer.start_training(train_data, test_data, classifier)
