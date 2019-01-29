import os
import pandas as pd
from torch.utils.data import Dataset


class TweetsDataset(Dataset):

    def __init__(self, root_dir, language_name, label_transform, tweet_transform=None):
        file_name = language_name + '.csv'
        file_path = os.path.join(root_dir, file_name)
        self.tweets_frame = pd.read_csv(file_path)

        self.label = label_transform(language_name)

        self.tweet_transform = tweet_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.tweets_frame)

    def __getitem__(self, idx):
        tweet = self.tweets_frame.iloc[idx]['tweet_text']
        if self.tweet_transform:
            tweet = self.tweet_transform(tweet)

        sample = {'tweet': tweet, 'label': self.label}

        return sample
