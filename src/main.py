from torchvision import transforms
from src.dataload.tweets_dataset import TweetsDataset
from src.dataload.tweet_transforms import *
from src.model.language_classifier import LanguageClassifier
import torch
import torch.nn as nn


transform = transforms.Compose(
    [SplitToWords(),
     RemoveURLs(),
     RemoveMentions(),
     RemoveResponseToken(),
     RemoveHashtags(),
     CleanToken(),
     RemoveBlanks(),
     ToLowerCase()])
data = TweetsDataset('../data', 'it', transform)
for i in range(10):
    print(data[i])


classifier = LanguageClassifier(num_of_classes=8)
pred = classifier(data[0]['tweet'])

loss = nn.CrossEntropyLoss()
target = torch.tensor([0])
output = loss(pred.unsqueeze(dim=0), target)
print(output)