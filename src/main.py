from src.dataload.tweets_dataset import TweetsDataset
from src.model.language_classifier import LanguageClassifier
import torch
import torch.nn as nn


data = TweetsDataset('../data', 'en')
print(data[0])


classifier = LanguageClassifier(num_of_classes=8)
pred = classifier(data[0]['tweet'])

loss = nn.CrossEntropyLoss()
target = torch.tensor([0])
output = loss(pred.unsqueeze(dim=0), target)
print(output)