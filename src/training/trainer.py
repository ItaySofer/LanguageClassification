import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataload.tweets_dataset import *


class Trainer:

    def __init__(self):
        self.logger = logging.getLogger('language_classifier')

    def start_training(self, model, tweets_data_handler, training_config):

        epochs = training_config['epochs']
        batch_size = training_config['batch_size']
        num_workers = training_config['num_workers']
        learning_rate = training_config['learning_rate']
        trained_models_output_root = training_config['trained_models_output_root']
        print_every_n_minibatches = 3   # TODO: Move to config, along with optimizer and criterion

        train_dataloader = tweets_data_handler.create_train_dataloader(batch_size=batch_size, num_workers=num_workers)
        test_dataloader = tweets_data_handler.create_test_dataloader(batch_size=batch_size, num_workers=num_workers)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.cuda()

        for epoch in range(1, epochs + 1):
            self.logger.info('Starting Epoch #%r -- TRAINING' % epoch)
            model = model.train()
            self.run_training_epoch(train_dataloader, model, optimizer, criterion, epoch, print_every_n_minibatches)

            self.save_model(model, trained_models_output_root, epoch)

            self.logger.info('Starting Epoch #%r -- EVALUATION' % epoch)
            model = model.eval()
            self.run_validation_epoch(test_dataloader, model, criterion, epoch, print_every_n_minibatches)

    def run_training_epoch(self, dataloader, model, optimizer, criterion, epoch, print_every_n_minibatches):

        running_loss = 0

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
                self.logger.info('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / print_every_n_minibatches))
                running_loss = 0.0

    def run_validation_epoch(self, dataloader, model, criterion, epoch, print_every_n_minibatches):

        calc_accuracy = lambda correct_preds, total_preds: 100.0 * correct_preds / total_preds

        with torch.no_grad():
            validation_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for i, sample_batched in enumerate(dataloader):
                tweets_data = sample_batched['tweet']
                labels = sample_batched['label']

                prediction_probs = model(tweets_data)

                # Calculate loss
                loss = criterion(prediction_probs, labels)
                validation_loss += loss.item()

                # Calculate accuracy
                predicted_labels = torch.argmax(input=prediction_probs.data, dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += len(tweets_data)

                if i % print_every_n_minibatches == print_every_n_minibatches - 1:
                    accuracy = calc_accuracy(correct_predictions, total_predictions)
                    self.logger.info('[%d, %5d] accuracy so far: %.3f' % (epoch, i + 1, accuracy))

            accuracy = calc_accuracy(correct_predictions, total_predictions)
            self.logger.info('Total validation accuracy: %.3f' % accuracy)

    @staticmethod
    def save_model(model, base_path, epoch):
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_path = os.path.join(base_path, 'classifier_%r.pt' % epoch)
        torch.save(model, model_path)
