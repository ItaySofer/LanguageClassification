import os
import pandas as pd
from torch.utils.data import Dataset


class TweetsDataset(Dataset):
    """
    A single language tweeter dataset.
    """

    def __init__(self, root_dir, language_name, label_transform, pass_original_tweet, tweet_transform=None):
        file_name = language_name + '.csv'
        file_path = os.path.join(root_dir, file_name)
        self.tweets_frame = pd.read_csv(file_path)

        self.language_name = language_name
        self.label = label_transform(language_name)

        self.tweet_transform = tweet_transform
        self.label_transform = label_transform
        self.pass_original_tweet = pass_original_tweet

    def __len__(self):
        return len(self.tweets_frame)

    def __getitem__(self, idx):
        original_tweet = self.tweets_frame.iloc[idx]['tweet_text']

        tweet = original_tweet
        if self.tweet_transform:
            tweet = self.tweet_transform(tweet)

        # tweet is expected to be of basestring type, otherwise if batching is used a customized collate_fn should
        # be specified.
        sample = {'tweet': tweet, 'label': self.label}

        if self.pass_original_tweet:
            sample.update({'original_tweet': original_tweet})

        return sample
