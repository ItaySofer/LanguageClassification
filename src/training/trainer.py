import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from src.dataload.tweets_dataset import *


class Trainer:

    def __init__(self):
        self.logger = logging.getLogger('language_classifier')

    def start_training(self, train_dataset, test_dataset, model, epochs=50):

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=filter_empty_collate_fn, num_workers=16)
        test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=filter_empty_collate_fn, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        base_path = '../trained_models'

        if torch.cuda.is_available():
            model = model.cuda()

        for epoch in range(1, epochs + 1):
            self.logger.info('Starting Epoch #%r -- TRAINING' % epoch)
            model = model.train()
            self.run_training_epoch(train_dataloader, model, optimizer, criterion, epoch)

            self.save_model(model, base_path, epoch)

            self.logger.info('Starting Epoch #%r -- EVALUATION' % epoch)
            model = model.eval()
            self.run_validation_epoch(test_dataloader, model, optimizer, criterion, epoch)

    def run_training_epoch(self, dataloader, model, optimizer, criterion, epoch):

        running_loss = 0
        print_every_n_minibatches = 3

        for i, sample_batched in enumerate(dataloader):
            tweets_data = sample_batched['tweet']
            labels = sample_batched['label']

            optimizer.zero_grad()
            pred = model(tweets_data)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % print_every_n_minibatches == print_every_n_minibatches - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i + 1, running_loss / print_every_n_minibatches))
                running_loss = 0.0

    def run_validation_epoch(self, dataloader, model, optimizer, criterion, epoch):
        # self.run_training_epoch(dataloader, model, optimizer, criterion, epoch)
        pass  # Skip for now..

    @staticmethod
    def save_model(model, base_path, epoch):
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_path = os.path.join(base_path, 'classifier_%r.pt' % epoch)
        torch.save(model, model_path)
