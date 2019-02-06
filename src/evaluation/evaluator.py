import logging
import torch
import torch.nn as nn

from src.evaluation.metrics import ExperimentMetrics


class Evaluator:
    _CRITERIONS = {
        'cross_entropy': nn.CrossEntropyLoss,
    }

    def __init__(self):
        self.logger = logging.getLogger('language_classifier')

    def start_evaluation(self, tweets_data_handler, model_path, training_config):
        self.logger.info('Beginning evaluation')
        self._evaluate(tweets_data_handler, model_path, training_config)

    def _evaluate(self, tweets_data_handler, model_path, training_config):
        batch_size = training_config['batch_size']
        num_workers = training_config['num_workers']
        metrics = training_config['test_data_metrics']
        criterion_type = training_config['criterion']

        model = torch.load(model_path)
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        dataloader = tweets_data_handler.create_test_dataloader(batch_size=batch_size, num_workers=num_workers)

        criterion_class = self._CRITERIONS[criterion_type]
        criterion = criterion_class()

        test_metrics = ExperimentMetrics(label_classes=tweets_data_handler.language_names, data_type='Test',
                                         metrics=metrics)

        self.run_evaluation(dataloader, model, criterion, test_metrics, tweets_data_handler.language_names)

    def run_evaluation(self, dataloader, model, criterion, metrics, language_names):
        misclassifications = {lang: [] for lang in language_names}

        with torch.no_grad():
            for i, sample_batched in enumerate(dataloader):
                tweets_data = sample_batched['tweet']
                labels = sample_batched['label']

                pred = model(tweets_data)
                loss = criterion(pred, labels)

                self._add_to_misclassifications(labels, pred, tweets_data, language_names,
                                                misclassifications)

                metrics.report_batch_results(epoch=1, preds=pred, labels=labels, loss=loss.item())

            self.logger.info('Finished evaluation')
            metrics.log_metrics(epoch=1)

            self.logger.info('Misclassifications in the form of { true language label : [misclassifications] } :')
            print(misclassifications)
            self.logger.info('The above line is in json format - copy it to a json viewer')


    @staticmethod
    def _add_to_misclassifications(labels, pred, tweets_data, language_names,
                                   wrong_predictions):
        pred_labels = torch.argmax(input=pred.data, dim=1)

        for i in range(len(tweets_data)):
            if pred_labels[i] != labels[i]:
                true_language_symbol = language_names[labels[i]]
                pred_language_symbol = language_names[pred_labels[i]]
                wrong_predictions[true_language_symbol].append(  # intentionaly appending a dictionary for json format printing
                    {"tweet": tweets_data[i],
                     "predicted": pred_language_symbol})

