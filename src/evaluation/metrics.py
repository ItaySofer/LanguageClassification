import sys
import logging
import numpy as np
import torch
from src.evaluation.metrics_decorators import handles_prediction, requires_confusion_matrix, metric


class ExperimentMetrics:
    """
    A class for aggregating metrics over training / evaluation process.
    """

    # All supported metrics & their titles
    metric_to_title = {
        'avg_loss': 'Average entry loss (per epoch)',
        'max_batch_loss': 'Maximum batch loss (per epoch)',
        'accuracy': 'Accuracy (%)',
        'avg_precision': 'Average Precision',
        'avg_recall': 'Average Recall',
        'avg_f1': 'Average F1 Score',
        'precision_per_class': 'Precision per class',
        'recall_per_class': 'Recall per class',
        'f1_per_class': 'F1 Score per class',
        'confusion_matrix': 'Confusion matrix (Ground Truth / Predictions)'
    }

    eps = sys.float_info.epsilon

    def __init__(self, label_classes, metrics=None, data_type='Training'):
        """
        :param label_classes: List of label names (each index corresponds to the label id as the model knows it)
        :param metrics: List of metrics to aggregate (keys of ExperimentMetrics.metric_to_title)
        :param data_type: 'Training' or 'Test'
        """
        self.logger = logging.getLogger('language_classifier')
        self.statistics_per_epoch = []
        self.label_classes = label_classes
        self.metrics = metrics or [metric for metric in ExperimentMetrics.metric_to_title.keys()]  # Default: all
        self.data_type = data_type

    def _fetch_epoch_statistics(self, epoch):
        """
        Get the current epoch aggregated statistics.
        If this is the first time the method is invoked for this epoch, the metrics entry will be initialized.
        :param epoch: Epoch number (one indexed)
        :return: Dictionary containing aggregated metrics for given epoch
        """
        if len(self.statistics_per_epoch) < epoch:
            num_classes = len(self.label_classes)
            epoch_statistics = {
                'entries_seen': 0,
                'correct_predictions': 0,
                'avg_loss': 0.0,
                'max_batch_loss': 0.0,
                'accuracy': 0.0,
                'avg_precision': 0.0,
                'avg_recall': 0.0,
                'avg_f1': 0.0,
                'precision_per_class': np.zeros(len(self.label_classes), dtype=float),
                'recall_per_class': np.zeros(len(self.label_classes), dtype=float),
                'f1_per_class': np.zeros(len(self.label_classes), dtype=float),
                'confusion_matrix': torch.zeros(num_classes, num_classes),
            }
            self.statistics_per_epoch.append(epoch_statistics)
        return self.statistics_per_epoch[epoch-1]

    @metric(name='avg_loss')
    def _report_avg_loss(self, epoch_statistics, loss, batch_size):
        entries_seen = epoch_statistics['entries_seen']
        running_avg_loss = epoch_statistics['avg_loss']
        total_entities = entries_seen + batch_size
        epoch_statistics['avg_loss'] = (running_avg_loss * entries_seen + loss * batch_size) / total_entities

    @metric(name='max_batch_loss')
    def _report_max_loss(self, epoch_statistics, loss):
        epoch_statistics['max_batch_loss'] = max(epoch_statistics['max_batch_loss'], loss)

    @metric(name='accuracy')
    def _report_accuracy(self, epoch_statistics, preds, labels, batch_size):
        predicted_labels = torch.argmax(input=preds.data, dim=1)
        correct_predictions = (predicted_labels == labels).sum().item()
        epoch_statistics['correct_predictions'] += correct_predictions
        total_entities = epoch_statistics['entries_seen'] + batch_size
        epoch_statistics['accuracy'] = 100.0 * epoch_statistics['correct_predictions'] / total_entities

    @metric(name='precision_per_class')
    @requires_confusion_matrix
    def _report_precision_per_class(self, epoch_statistics):
        confusion_matrix = epoch_statistics['confusion_matrix']
        per_class_precision = confusion_matrix.diag() / (confusion_matrix.sum(1) + self.eps)
        epoch_statistics['precision_per_class'] = per_class_precision.numpy()

    @metric(name='recall_per_class')
    @requires_confusion_matrix
    def _report_recall_per_class(self, epoch_statistics):
        confusion_matrix = epoch_statistics['confusion_matrix']
        per_class_recall = confusion_matrix.diag() / (confusion_matrix.sum(0) + self.eps)
        epoch_statistics['recall_per_class'] = per_class_recall.numpy()

    @metric(name='f1_per_class')
    @requires_confusion_matrix
    def _report_f1_per_class(self, epoch_statistics):
        confusion_matrix = epoch_statistics['confusion_matrix']
        per_class_precision = confusion_matrix.diag() / (confusion_matrix.sum(1) + self.eps)
        per_class_recall = confusion_matrix.diag() / (confusion_matrix.sum(0) + self.eps)
        per_class_f1 = (2 * per_class_precision * per_class_recall) / (per_class_precision + per_class_recall + self.eps)
        epoch_statistics['f1_per_class'] = per_class_f1.numpy()

    @metric(name='avg_precision')
    @requires_confusion_matrix
    def _report_avg_precision(self, epoch_statistics):
        confusion_matrix = epoch_statistics['confusion_matrix']
        per_class_precision = confusion_matrix.diag() / (confusion_matrix.sum(1) + self.eps)
        avg_precision = torch.mean(per_class_precision)
        epoch_statistics['avg_precision'] = avg_precision.numpy()

    @metric(name='avg_recall')
    @requires_confusion_matrix
    def _report_avg_recall(self, epoch_statistics):
        confusion_matrix = epoch_statistics['confusion_matrix']
        per_class_recall = confusion_matrix.diag() / (confusion_matrix.sum(0) + self.eps)
        avg_recall = torch.mean(per_class_recall)
        epoch_statistics['avg_recall'] = avg_recall.numpy()

    @metric(name='avg_f1')
    @requires_confusion_matrix
    def _report_avg_f1(self, epoch_statistics):
        confusion_matrix = epoch_statistics['confusion_matrix']
        per_class_precision = confusion_matrix.diag() / (confusion_matrix.sum(1) + self.eps)
        per_class_recall = confusion_matrix.diag() / (confusion_matrix.sum(0) + self.eps)
        per_class_f1 = (2 * per_class_precision * per_class_recall) / (per_class_precision + per_class_recall + self.eps)
        avg_f1 = torch.mean(per_class_f1)
        epoch_statistics['avg_f1'] = avg_f1.numpy()

    def _report_total_processed(self, epoch_statistics, batch_size):
        epoch_statistics['entries_seen'] += batch_size

    def _update_confusion_matrix(self, epoch_statistics, labels, preds):
        """
        Updates the confusion matrix of the current predictions vs labels.
        E.g: A matrix of num_classes x num_classes which describes the amount of times each prediction was mistaken
        for another label. The matrix appears like so:

                            Prediction
                          ----------------------
                          | 0 | 1 | .... | C-1 |
                    ----------------------------
       Ground Truth | 0   |   |   | .... |     |
                    ----------------------------
                    | 1   |   |   | .... |     |
                    ----------------------------
                    | ... |  ...............   |
                    ----------------------------
                    | C-1 |   |   | .... |     |
                    ----------------------------

        See: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/

        :param labels: Ground truth labels for the current batch
        :param preds: Predictions of the model for the current batch
        :return: Confusion matrix as PyTorch Tensor: (num_classes x num_classes)
        """
        with torch.no_grad():
            predicted_labels = torch.argmax(input=preds.data, dim=1)
            confusion_matrix = epoch_statistics['confusion_matrix']
            for t, p in zip(labels.view(-1), predicted_labels.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    @handles_prediction
    def report_batch_results(self, epoch, preds, labels, loss):
        epoch_statistics = self._fetch_epoch_statistics(epoch)
        batch_size = labels.shape[0]

        self._report_avg_loss(epoch_statistics, loss, batch_size)
        self._report_max_loss(epoch_statistics, loss)
        self._report_accuracy(epoch_statistics, preds, labels, batch_size)
        self._report_precision_per_class(epoch_statistics)
        self._report_recall_per_class(epoch_statistics)
        self._report_f1_per_class(epoch_statistics)
        self._report_avg_precision(epoch_statistics)
        self._report_avg_recall(epoch_statistics)
        self._report_avg_f1(epoch_statistics)
        self._report_total_processed(epoch_statistics, batch_size)

    def log_metrics(self, epoch):
        np.set_printoptions(precision=3)
        epoch_statistics = self._fetch_epoch_statistics(epoch)
        self.logger.info('Metrics for %r - epoch #%r:' % (self.data_type, epoch))
        self.logger.info('-------------------------')
        self.logger.info('- Entries processed: %r' % epoch_statistics['entries_seen'])

        for metric in self.metrics:
            metric_title = self.metric_to_title[metric]
            metric_value = epoch_statistics[metric]
            if hasattr(metric_value, 'size'):
                if isinstance(metric_value.size, int):
                    if metric_value.size > 1:
                        self.logger.info('- %r: %r' % (metric_title, str(metric_value)))
                    else:
                        self.logger.info('- %r: %.3f' % (metric_title, metric_value))
                elif len(metric_value.size()) == 2:
                    self.logger.info('- %r:\n %r' % (metric_title, metric_value))
            else:
                self.logger.info('- %r: %.3f' % (metric_title, metric_value))

    def __getitem__(self, epoch):
        return self.statistics_per_epoch[epoch - 1]

    def __iter__(self):
        return iter(self.statistics_per_epoch)
