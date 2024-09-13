import sys
from datetime import datetime

import pandas as pd
import numpy as np
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from scipy.stats import ks_2samp
import joblib


class DataAugmentation:
    """
    Class to perform data augmentation using CTGAN model

    Uses Bayesian optimization to find the best hyperparameters for the CTGAN model

    Usage
    ----------
    >>> data_path = 'data.csv'
    >>> num_columns = ['RSSI A', 'RSSI B', 'RSSI C']
    >>> cat_columns = ['x', 'y', 'RP']

    >>> trainer = DataAugmentation(data_path, num_columns, cat_columns)
    >>> trainer.optimize_parameters(n_calls=10,verbose=True)
    >>> trainer.train_model()
    >>> trainer.save_model('model_optimized.pkl')

    >>> trainer.load_model('model_optimized.pkl')
    >>> synthetic_data = trainer.generate_data(260)
    >>> synthetic_data.to_csv('synthetic_data_optimized.csv', index=False)

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the data

    num_columns : list
        List of column names of numerical columns in the data

    cat_columns : list
        List of column names of categorical columns in the data

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame containing the data

    num_columns : list
        List of column names of numerical columns in the data

    cat_columns : list
        List of column names of categorical columns in the data

    model : RegularSynthesizer
        Instance of RegularSynthesizer model

    best_params : list
        Best hyperparameters found by Bayesian optimization

    best_score : float
        Best score achieved by the model
    """

    def __init__(self, data_path, num_columns, cat_columns):
        if data_path != "":
            self.data = pd.read_csv(data_path)
        self.num_columns = num_columns
        self.cat_columns = cat_columns

        self.model = None

        self.best_params = None
        self.best_score = None

    def results(self):
        print(f"{datetime.now().strftime('%d-%m-%y_%H:%M:%S')}")
        print(f"Best parameters: {self.best_params}")
        print(f"Best score: {self.best_score}")

    def evaluate_model(self, params):
        batch_size, learning_rate, epochs, beta_1, beta_2 = params

        ctgan_args = ModelParameters(batch_size=int(batch_size),
                                     lr=learning_rate,
                                     betas=(beta_1, beta_2))

        train_args = TrainParameters(epochs=int(epochs))

        synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
        synth.fit(data=self.data, train_arguments=train_args, num_cols=self.num_columns, cat_cols=self.cat_columns)

        synthetic_data = synth.sample(len(self.data))

        ks_scores = []
        for column in self.num_columns:
            ks_stat, _ = ks_2samp(self.data[column], synthetic_data[column])
            ks_scores.append(ks_stat)
        mean_ks_score = np.mean(ks_scores)
        print(f"Mean KS score: {mean_ks_score}", file=sys.stderr)
        return -mean_ks_score

    def optimize_parameters(self, n_calls=20, verbose=False):
        search_space = [
            Categorical([x for x in range(100, 501, 10)], name='batch_size'),
            Real(1e-5, 1e-3, prior='log-uniform', name='learning_rate'),
            Integer(400, 1500, name='epochs'),
            Real(0.5, 0.599, name='beta_1'),
            Real(0.9, 0.999, name='beta_2')
        ]

        start_time = datetime.now()
        res = gp_minimize(self.evaluate_model, search_space, n_calls=n_calls, random_state=42, verbose=verbose)
        end_time = datetime.now()
        print(f"Time taken: {end_time - start_time}", file=sys.stderr)

        self.best_params = res.x
        self.best_score = res.fun

        self.results()

    def train_model(self):
        best_batch_size, best_learning_rate, best_epochs, best_beta_1, best_beta_2 = self.best_params

        ctgan_args = ModelParameters(batch_size=int(best_batch_size),
                                     lr=best_learning_rate,
                                     betas=(best_beta_1, best_beta_2))

        train_args = TrainParameters(epochs=int(best_epochs))
        self.model = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
        self.model.fit(data=self.data, train_arguments=train_args, num_cols=self.num_columns, cat_cols=self.cat_columns)

    def _save_params(self, filename):
        joblib.dump(self.best_params, filename)

    def save_model(self, filename):
        self.model.save(filename)
        self._save_params(f"params_{filename}")

    def load_model(self, filename):
        self.model = RegularSynthesizer.load(filename)

    def generate_data(self, n_samples):
        if self.model is None:
            raise ValueError("Model not loaded or trained!")
        return self.model.sample(n_samples)
