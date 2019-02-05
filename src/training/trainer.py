import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataload.tweets_dataset import *
from src.evaluation.metrics import ExperimentMetrics
from src.evaluation.plotter import Plotter


class Trainer:

    _CRITERIONS = {
        'cross_entropy': nn.CrossEntropyLoss,
    }

    _OPTIMIZERS = {
        'adam': optim.Adam,
        'sgd': optim.SGD
    }

    def __init__(self, experiment_name='Default'):
        self.logger = logging.getLogger('language_classifier')
        self.experiment_name = experiment_name
        self.plotter = Plotter(experiment_env=experiment_name)

    def _train(self, model, tweets_data_handler, training_config, trainer_state=None):
        epochs = training_config['epochs']
        batch_size = training_config['batch_size']
        num_workers = training_config['num_workers']
        learning_rate = training_config['learning_rate']
        trained_models_output_root = training_config['trained_models_output_root']
        print_every_n_minibatches = training_config['log_rate']

        train_dataloader = tweets_data_handler.create_train_dataloader(batch_size=batch_size, num_workers=num_workers)
        test_dataloader = tweets_data_handler.create_test_dataloader(batch_size=batch_size, num_workers=num_workers)

        criterion_class = self._CRITERIONS[training_config['criterion']]
        criterion = criterion_class()

        if trainer_state is None:
            optimizer_class = self._OPTIMIZERS[training_config['optimizer']]
            optimizer = optimizer_class(model.parameters(), lr=learning_rate)
            training_metrics = ExperimentMetrics(label_classes=tweets_data_handler.language_names, data_type='Training',
                                                 metrics=training_config['train_data_metrics'])
            test_metrics = ExperimentMetrics(label_classes=tweets_data_handler.language_names, data_type='Test',
                                             metrics=training_config['test_data_metrics'])
            first_epoch = 1
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            optimizer_state = trainer_state['optimizer']
            optimizer.load_state_dict(optimizer_state)
            training_metrics = trainer_state['training_metrics']
            test_metrics = trainer_state['test_metrics']
            first_epoch = trainer_state['epoch']

            # Redraw all plots for past epochs
            for past_epoch in range(1, first_epoch):
                self.plotter.plot_aggregated_metrics(metrics=training_metrics, epoch=past_epoch)
                self.plotter.plot_aggregated_metrics(metrics=test_metrics, epoch=past_epoch)

        if torch.cuda.is_available():
            model = model.cuda()

        for epoch in range(first_epoch, epochs + 1):
            model = model.train()
            self.run_training_epoch(train_dataloader, model,
                                    optimizer, criterion, training_metrics,
                                    epoch, print_every_n_minibatches)
            self.save_model(model, trained_models_output_root, epoch)
            self.save_trainer_state(trained_models_output_root, optimizer, training_metrics, test_metrics, epoch)
            model = model.eval()
            self.run_validation_epoch(test_dataloader, model,
                                      criterion, test_metrics,
                                      epoch, print_every_n_minibatches)

    def start_training(self, model, tweets_data_handler, training_config):
        self.logger.info('Beginning training from blank slate.')
        self._train(model, tweets_data_handler, training_config)

    def resume_training(self, tweets_data_handler, training_config, model_path, trainer_state_path):
        self.logger.info('Resuming training from: %r' % model_path)
        model = torch.load(model_path)
        trainer_state = torch.load(trainer_state_path)
        self._train(model, tweets_data_handler, training_config, trainer_state)

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
        last_model_path = os.path.join(base_path, 'last_model.pt')
        torch.save(model, last_model_path)
        self.logger.info('Trained classifier saved at: %r' % model_path)

    def save_trainer_state(self, base_path, optimizer, training_metrics, test_metrics, epoch):
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        trainer_state = {
            'optimizer': optimizer.state_dict(),
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'epoch': epoch
        }

        trainer_state_path = os.path.join(base_path, 'last_trainer_state.pt')
        torch.save(trainer_state, trainer_state_path)
        self.logger.info('Trained classifier saved at: %r' % trainer_state_path)
