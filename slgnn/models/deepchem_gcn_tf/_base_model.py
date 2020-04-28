import os
import csv
import tempfile
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from .utils.save import log


class Model(BaseEstimator, metaclass=ABCMeta):
    """ Abstract base class that supports other ML models
    """

    def __init__(self,
                 model_instance=None,
                 model_dir=None,
                 verbose=True):
        """Abstract class for all models.
        Parameters:
        -----------
        model_instance (object): Wrapper around ScikitLearn/Keras/Tensorflow
            model object.
        model_dir (str): Path to directory where model will be stored.
        verbose (bool): Boolean.
        """
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            model_dir = self.temp_dir.name
        self.model_dir = model_dir
        self.model_instance = model_instance
        self.model_class = model_instance.__class__

        self.verbose = verbose

    def __del__(self):
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    @abstractmethod
    def fit_on_batch(self):
        """ Updates existing model with new information.
        """

    @abstractmethod
    def predict_on_batch(self):
        """ Makes predictions on given batch of new data.

        Parameters
        ----------
        X: np.ndarray
        Features
        """

    @abstractmethod
    def reload(self):
        """ Reload trained model from disk.
        """

    @staticmethod
    def get_model_filename(model_dir):
        """ Given model directory, obtain filename for the model itself.
        """
        return os.path.join(model_dir, "model.joblib")

    @staticmethod
    def get_params_filename(model_dir):
        """ Given model directory, obtain filename for the model itself.
        """
        return os.path.join(model_dir, "model_params.joblib")

    @abstractmethod
    def save(self):
        """ Dispatcher function for saving.
        """

    def fit(self, dataset, nb_epoch=10, batch_size=32, **kwargs):
        """ Fits a model on data in a Dataset object.
        """
        for epoch in range(nb_epoch):
            log("Starting epoch {}".format(epoch+1), self.verbose)
            losses = list()
            for (X_batch, y_batch, w_batch,
                 ids_batch) in dataset.iterbatches(batch_size):
                losses.append(self.fit_on_batch(X_batch, y_batch, w_batch))
                log("Avg loss for epoch {}: {}".format(
                    epoch+1, np.array(losses).mean()),
                    verbose=self.verbose)

    def predict(self, dataset, batch_size=None):
        """ Uses self to make predictions on provided Dataset object.

        Returns:
        y_pred: numpy ndarray of shape (n_samples,)
        """
        y_preds = list()

        for (X_batch, _, _,
             ids_batch) in dataset.iterbatches(batch_size, deterministic=True):
            n_samples = len(X_batch)
            y_pred_batch = self.predict_on_batch(X_batch)
            y_pred_batch = y_pred_batch[:n_samples]
            y_preds.append(y_pred_batch)
        y_pred = np.concatenate(y_preds)
        return y_pred

    def evaluate(self, dataset, metrics, per_task_metrics=False):
        """ Evaluates the performance of this model on specified dataset.

        Args:
        ----------
        dataset (dc.data.Dataset): Dataset object.
        metrics (deepchem.metrics.Metric: Evaluation metric.
        per_task_metrics (bool): If True, return per-task scores.

        Returns:
        -------
        scores (dict): Maps tasks to scores under metric.
        per_task_scores (dict): If per_task_metrics is True. Maps per task
            scores under metric.
        """
        evaluator = Evaluator(self, dataset, verbose=self.verbose)
        if not per_task_metrics:
            scores = evaluator.compute_model_performance(metrics)
            return scores
        else:
            scores, per_task_scores = evaluator.compute_model_performance(
                metrics, per_task_metrics=per_task_metrics)
            return scores, per_task_scores

    @abstractmethod
    def get_task_type(self):
        """ Currently models can only be classifiers or regressors.
        """

    @abstractmethod
    def get_num_tasks(self):
        """ Get number of tasks.
        """


class Evaluator(object):
    """ Class that evaluates a model on a given dataset."""

    def __init__(self, model, dataset, verbose=False):
        self.model = model
        self.dataset = dataset
        self.task_names = dataset.get_task_names()
        self.verbose = verbose

    def output_statistics(self, scores, stats_out):
        """ Write computed stats to file.
        """
        with open(stats_out, "w") as statsfile:
            statsfile.write(str(scores) + "\n")

    def output_predictions(self, y_preds, csv_out):
        """ Writes predictions to file.

        Args:
        -------
        y_preds (np.ndarray)
        csv_out (Open file object)
        """
        mol_ids = self.dataset.ids
        n_tasks = len(self.task_names)
        y_preds = np.reshape(y_preds, (len(y_preds), n_tasks))
        assert len(y_preds) == len(mol_ids)
        with open(csv_out, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Compound"] + self.dataset.get_task_names())
            for mol_id, y_pred in zip(mol_ids, y_preds):
                csvwriter.writerow([mol_id] + list(y_pred))

    def compute_model_performance(self,
                                  metrics,
                                  csv_out=None,
                                  stats_out=None,
                                  per_task_metrics=False):
        """ Computes statistics of model on test data and saves results to csv.

        Args:
        ----------
        metrics (list): List of dc.metrics.Metric objects.
        csv_out (str): Optional. Filename to write CSV of model predictions.
        stats_out (str): Optional. Filename to write computed statistics.
        per_task_metrics (bool): Optional. If true, return computed metric for
            each task on multitask dataset.
        """
        y = self.dataset.y
        w = self.dataset.w

        if not len(metrics):
            return {}
        else:
            mode = metrics[0].mode
        y_pred = self.model.predict(self.dataset, self.output_transformers)
        if mode == "classification":
            y_pred_print = np.argmax(y_pred, -1)
        else:
            y_pred_print = y_pred
        multitask_scores = {}
        all_task_scores = {}

        if csv_out is not None:
            log("Saving predictions to %s" % csv_out, self.verbose)
            self.output_predictions(y_pred_print, csv_out)

        # Compute multitask metrics
        for metric in metrics:
            if per_task_metrics:
                multitask_scores[metric.name], computed_metrics = \
                    metric.compute_metric(y, y_pred, w, per_task_metrics=True)
                all_task_scores[metric.name] = computed_metrics
            else:
                multitask_scores[metric.name] = metric.compute_metric(
                    y, y_pred, w, per_task_metrics=False)

        if stats_out is not None:
            log("Saving stats to %s" % stats_out, self.verbose)
            self.output_statistics(multitask_scores, stats_out)

        if not per_task_metrics:
            return multitask_scores
        else:
            return multitask_scores, all_task_scores
