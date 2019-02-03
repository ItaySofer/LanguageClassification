import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


class Evaluator:

    def __init__(self):
        self.logger = logging.getLogger('language_classifier')

    def start_evaluation(self, train_dataset, test_dataset, model_path):

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        model = self.load_model(model_path)

        if torch.cuda.is_available():
            model = model.cuda()

        model = model.eval()
        criterion = nn.CrossEntropyLoss()

        self.logger.info('Starting Evaluation on training data:')
        self.run_evaluation_on_data(train_dataloader, model, criterion)

        self.logger.info('Starting Evaluation on validation data:')
        self.run_evaluation_on_data(test_dataloader, model, criterion)

    def run_evaluation_on_data(self, dataloader, model, criterion):

        calc_accuracy = lambda correct_preds, total_preds: 100.0 * correct_preds / total_preds

        with torch.no_grad():
            validation_loss = 0
            correct_predictions = 0
            total_predictions = 0
            print_every_n_minibatches = 3

            for i, sample_batched in enumerate(dataloader):
                tweets_data = sample_batched['tweet']
                labels = sample_batched['label']

                prediction_probs = model(tweets_data)

                # Calculate avg_loss
                loss = criterion(prediction_probs, labels)
                validation_loss += loss.item()

                # Calculate accuracy
                predicted_labels = torch.argmax(input=prediction_probs.data, dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += len(tweets_data)

                if i % print_every_n_minibatches == print_every_n_minibatches - 1:
                    accuracy = calc_accuracy(correct_predictions, total_predictions)
                    self.logger.info('[%5d] accuracy so far: %.3f' % (i + 1, accuracy))

            accuracy = calc_accuracy(correct_predictions, total_predictions)
            self.logger.info('Total validation accuracy: %.3f' % accuracy)

    @staticmethod
    def load_model(model_path):
        return torch.load(model_path)
