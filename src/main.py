from torchvision import transforms
from src.dataload.tweets_dataset import TweetsDataset
from src.dataload.tweet_transforms import *
from src.dataload.label_transforms import ToOneHot
from src.model.language_classifier import LanguageClassifier
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset


def prepare_data():
    language_names = ['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl']

    tweet_transform = transforms.Compose(
        [SplitToWords(),
         RemoveURLs(),
         RemoveMentions(),
         RemoveResponseToken(),
         RemoveHashtags(),
         CleanToken(),
         RemoveBlanks(),
         ToLowerCase()])
    label_transform = ToOneHot(language_names=language_names)

    language_datasets = []
    for language_name in language_names:
        dataset = TweetsDataset(root_dir='../data', language_name=language_name,
                                tweet_transform=tweet_transform, label_transform=label_transform)
        language_datasets.append(dataset)

    return ConcatDataset(language_datasets)


data = prepare_data()

classifier = LanguageClassifier(num_of_classes=8)
pred = classifier(data[0]['tweet'])

loss = nn.CrossEntropyLoss()
target = torch.tensor([0])
output = loss(pred.unsqueeze(dim=0), target)
print(output)