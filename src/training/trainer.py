import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataload.tweets_dataset import *
from src.evaluation.metrics import ExperimentMetrics
from src.evaluation.plotter import Plotter


class Trainer:

    def __init__(self, experiment_name='Default'):
        self.logger = logging.getLogger('language_classifier')
        self.experiment_name = experiment_name
        self.plotter = Plotter(experiment_env=experiment_name)

    def start_training(self, model, tweets_data_handler, training_config):

        epochs = training_config['epochs']
        batch_size = training_config['batch_size']
        num_workers = training_config['num_workers']
        learning_rate = training_config['learning_rate']
        trained_models_output_root = training_config['trained_models_output_root']
        print_every_n_minibatches = 200   # TODO: Move to config, along with optimizer and criterion

        train_dataloader = tweets_data_handler.create_train_dataloader(batch_size=batch_size, num_workers=num_workers)
        test_dataloader = tweets_data_handler.create_test_dataloader(batch_size=batch_size, num_workers=num_workers)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        training_metrics = ExperimentMetrics(label_classes=tweets_data_handler.language_names, data_type='Training',
                                             metrics=['avg_loss', 'accuracy',
                                                      'avg_precision', 'avg_recall', 'avg_f1',
                                                      'confusion_matrix'])
        test_metrics = ExperimentMetrics(label_classes=tweets_data_handler.language_names, data_type='Test',
                                         metrics=['avg_loss', 'accuracy',
                                                  'avg_precision', 'avg_recall', 'avg_f1',
                                                  'precision_per_class', 'recall_per_class', 'f1_per_class',
                                                  'confusion_matrix']
                                         )

        if torch.cuda.is_available():
            model = model.cuda()

        for epoch in range(1, epochs + 1):
            model = model.train()
            self.run_training_epoch(train_dataloader, model,
                                    optimizer, criterion, training_metrics,
                                    epoch, print_every_n_minibatches)
            self.save_model(model, trained_models_output_root, epoch)
            model = model.eval()
            self.run_validation_epoch(test_dataloader, model,
                                      criterion, test_metrics,
                                      epoch, print_every_n_minibatches)

    def run_training_epoch(self, dataloader, model, optimizer, criterion, metrics, epoch, print_every_n_minibatches):

        self.logger.info('Starting Epoch #%r -- TRAINING' % epoch)

        for i, sample_batched in enumerate(dataloader):
            tweets_data = sample_batched['tweet']
            labels = sample_batched['label']

            optimizer.zero_grad()
            pred = model(tweets_data)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            metrics.report_batch_results(epoch=epoch, preds=pred, labels=labels, loss=loss.item())

            if i % print_every_n_minibatches == print_every_n_minibatches - 1:
                metrics.log_metrics(epoch)

        self.logger.info('Finished Epoch #%r -- TRAINING' % epoch)
        metrics.log_metrics(epoch)
        self.plotter.plot_aggregated_metrics(metrics=metrics, epoch=epoch)

    def run_validation_epoch(self, dataloader, model, criterion, metrics, epoch, print_every_n_minibatches):

        with torch.no_grad():
            for i, sample_batched in enumerate(dataloader):
                tweets_data = sample_batched['tweet']
                labels = sample_batched['label']

                pred = model(tweets_data)
                loss = criterion(pred, labels)

                metrics.report_batch_results(epoch=epoch, preds=pred, labels=labels, loss=loss.item())

                if i % print_every_n_minibatches == print_every_n_minibatches - 1:
                    metrics.log_metrics(epoch)

            self.logger.info('Finished Epoch #%r -- EVALUATION' % epoch)
            metrics.log_metrics(epoch)
            self.plotter.plot_aggregated_metrics(metrics=metrics, epoch=epoch)

    def save_model(self, model, base_path, epoch):
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_path = os.path.join(base_path, 'classifier_%r.pt' % epoch)
        torch.save(model, model_path)
        self.logger.info('Trained classifier saved at: %r' % model_path)
