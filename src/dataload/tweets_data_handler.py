from src.dataload.tweets_dataset import TweetsDataset
from torch.utils.data import ConcatDataset, Subset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from src.dataload.tweet_transforms import *
from src.dataload.label_transforms import *


class TweetsDataHandler:
    """
    A facade for handling all related Twitter multi-langual data services.
    Data is loaded through this interface, and dataloaders should be created using its helper functions.
    """

    def __init__(self, language_names, data_root, train_test_split_perc):
        """
        Creates a new handler instance.
        Creating a new TweetsDataHandler invokes the loading process of training & test datasets, which are cached
        for further uses.
        Therefore TweetsDataHandler is a heavyweight object.
        :param language_names: List of language initials whose data will be loaded.
        :param data_root: Path of the root data folder, where the twitter *.csv files reside.
        :param train_test_split_perc: Percentage of train / test data split (in range of 0.0-1.0 where the number
        represents the train percentage).
        """
        self.tweet_transform = transforms.Compose(
            [SplitToWords(),
             RemoveURLs(),
             RemoveMentions(),
             RemoveResponseToken(),
             RemoveHashtags(),
             RemoveNames(),
             CleanTokens(),
             RemoveBlanks(),
             ToLowerCase(),
             JoinWordsToSentence()])

        self.label_transform = transforms.Compose(
            [ToLabelIdTensor(language_names=language_names),
             ToCuda()])

        self.language_names = language_names
        self.train_data, self.test_data = self._prepare_data(data_root=data_root,
                                                             train_percent=train_test_split_perc)

    def _prepare_data(self, data_root, train_percent):
        """
        This method:
        1) Loads each of the language tweet-datasets.
        2) Splits each dataset to train / test subsets, according to train_percent.
        3) Concatenates all train datasets and all test datasets respectively to create a pair of multilingual datasets.
        :param data_root: Path of the root data folder, where the twitter *.csv files reside.
        :param train_test_split_perc: Percentage of train / test data split (in range of 0.0-1.0 where the number
        :return: Train & test Datasets (ConcatDataset of multiple subsets of TweetsDataset)
        """
        train_datasets = []
        test_datasets = []
        for language_name in self.language_names:
            dataset = TweetsDataset(root_dir=data_root, language_name=language_name,
                                    tweet_transform=self.tweet_transform, label_transform=self.label_transform)

            train_size = int(train_percent * len(dataset))
            train_dataset = Subset(dataset, range(0, train_size))
            test_dataset = Subset(dataset, range(train_size, len(dataset)))

            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)

        return ConcatDataset(train_datasets), ConcatDataset(test_datasets)

    def create_train_dataloader(self, batch_size, num_workers):
        """
        Create a new dataloader for pre-loaded training data.
        :param batch_size: Batch size returned by each iteration of the dataloader.
        :param num_workers: Number of sub-processes to employ for pre-fetch.
        :return: Pytorch DataLoader that wraps the training data.
        """
        train_dataloader = DataLoader(self.train_data,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      collate_fn=self.filter_empty_collate_fn,
                                      num_workers=num_workers)
        return train_dataloader

    def create_test_dataloader(self, batch_size, num_workers):
        """
        Create a new dataloader for pre-loaded test data.
        :param batch_size: Batch size returned by each iteration of the dataloader.
        :param num_workers: Number of sub-processes to employ for pre-fetch.
        :return: Pytorch DataLoader that wraps the test data.
        """
        test_dataloader = DataLoader(self.test_data,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     collate_fn=self.filter_empty_collate_fn,
                                     num_workers=num_workers)
        return test_dataloader

    @staticmethod
    def filter_empty_collate_fn(batch):
        """ Tweeter dataloaders override the default collate function to make sure empty entries are not returned
        (this case may arise for some tweets after transformations are applied
        """
        batch = list(filter(lambda sample: sample['tweet'] is not '', batch))
        return default_collate(batch)
