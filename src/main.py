from torchvision import transforms
from src.dataload.tweets_dataset import TweetsDataset
from src.dataload.tweet_transforms import *
from src.dataload.label_transforms import ToOneHot
from src.model.language_classifier import LanguageClassifier
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Subset, DataLoader


def prepare_data(language_names, tweet_transform, label_transform, train_percent):
    train_datasets = []
    test_datasets = []
    for language_name in language_names:
        dataset = TweetsDataset(root_dir='../data', language_name=language_name,
                                tweet_transform=tweet_transform, label_transform=label_transform)

        train_size = int(train_percent * len(dataset))
        train_dataset = Subset(dataset, range(0, train_size))
        test_dataset = Subset(dataset, range(train_size, len(dataset)))

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    return ConcatDataset(train_datasets), ConcatDataset(test_datasets)


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
train_data, test_data = prepare_data(language_names, tweet_transform, label_transform, train_percent=0.7)


# train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
#
# dataloaders = [train_dataloader, test_dataloader]
# for dataloader in dataloaders:
#     for i, sample_batched in enumerate(dataloader):
#         print(sample_batched)
#         if i == 2:
#             break

classifier = LanguageClassifier(num_of_classes=8)
pred = classifier(train_data[0]['tweet'])

loss = nn.CrossEntropyLoss()
target = torch.tensor([0])
output = loss(pred.unsqueeze(dim=0), target)
print(output)