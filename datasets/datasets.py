import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class Dataset:
    def __init__(self, data, treatment_column, outcome_column, covariate_columns, truth=None, folds=None):
        self.data = data
        self.treatment_column = treatment_column
        self.outcome_column = outcome_column
        self.covariate_columns = covariate_columns
        self.truth = truth
        self.folds = folds

    @property
    def n_covariates(self):
        return len(self.covariate_columns)

    def to_batch(self):
        if self.treatment_column is not None:
            return (
                self.covariates_tensor(),
                self.treatments_tensor(),
                self.outcomes_tensor(),
            )
        else:
            return (
                self.covariates_tensor(),
                self.outcomes_tensor(),
            )

    def create_dataloader(self, batch_size):
        dataset = TensorDataset(*self.to_batch())
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    def outcomes_tensor(self):
        outcomes = self.data[:, self.outcome_column].astype(np.float32)
        return torch.from_numpy(outcomes).reshape(-1, 1)

    def treatments_tensor(self):
        treatments = self.data[:, self.treatment_column].astype(np.float32)
        return torch.from_numpy(treatments).reshape(-1, 1)

    def covariates_tensor(self):
        covariates = self.data[:, self.covariate_columns].astype(np.float32)
        return torch.from_numpy(covariates)

    def _with_data(self, data, folds=None):
        dataset = copy.copy(self)
        dataset.data = data
        dataset.folds = folds
        return dataset

    def standardize_covariates(self, mean=None, std=None):
        if mean is None and std is None:
            covariates = self.data[:, self.covariate_columns]
            mean = covariates.mean(axis=0)
            std = covariates.std(axis=0)
        self.data[:, self.covariate_columns] = (self.data[:, self.covariate_columns] - mean) / std
        return mean, std

    def bootstrap(self):
        bootstrap_indices = np.random.choice(self.data.shape[0], size=self.data.shape[0], replace=True)
        return self._with_data(self.data[bootstrap_indices, :])

    def get_fit_and_test_folds(self, test_fold_index):
        fit_folds = [self.folds[i] for i in range(len(self.folds)) if i != test_fold_index]
        fit_fold_indices = np.concat(fit_folds)
        test_fold_indices = self.folds[test_fold_index]
        return (
            self._with_data(self.data[fit_fold_indices, :]),
            self._with_data(self.data[test_fold_indices, :]),
        )

    def split_into_train_and_validation_sets(self, train_size):
        train_data, test_data = train_test_split(self.data, train_size=train_size)
        return (
            self._with_data(train_data),
            self._with_data(test_data),
        )

    def create_folds(self, n_folds):
        number_of_samples = self.data.shape[0]
        indices = np.arange(number_of_samples, dtype=int)
        np.random.shuffle(indices)
        self.folds = [fold for fold in np.array_split(indices, n_folds)]


class ATEDataset(Dataset):
    def __init__(self, data, treatment_column, outcome_column, covariate_columns, truth=None, folds=None):
        super().__init__(data, treatment_column, outcome_column, covariate_columns, truth, folds)

    def split_into_train_and_validation_sets(self, train_size):
        train_data, test_data = train_test_split(
            self.data, train_size=train_size, stratify=self.data[:, self.treatment_column]
        )
        return (
            self._with_data(train_data),
            self._with_data(test_data),
        )

    def create_folds(self, n_folds):
        number_of_samples = self.data.shape[0]
        indices = np.arange(number_of_samples, dtype=int)
        treated_indices = indices[self.data[:, self.treatment_column] == 1]
        control_indices = indices[self.data[:, self.treatment_column] == 0]
        np.random.shuffle(treated_indices)
        np.random.shuffle(control_indices)
        treated_fold_indices = np.array_split(treated_indices, n_folds)
        control_fold_indices = np.array_split(control_indices, n_folds)
        self.folds = [
            np.concat((treated, control)) for treated, control in zip(treated_fold_indices, control_fold_indices)
        ]


class IHDPDataset(ATEDataset):
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        super().__init__(data, treatment_column, outcome_column, covariate_columns)
        self.truth = np.mean(self.data[:, 4] - self.data[:, 3])

    @classmethod
    def load_replication(cls, replication_id):
        path = "datasets/ihdp_replications/ihdp_" + str(replication_id) + ".csv"
        data = np.loadtxt(path)
        return cls(data=data, treatment_column=0, outcome_column=1, covariate_columns=[i + 5 for i in range(25)])


class ToyExampleDataset(ATEDataset):
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        super().__init__(data, treatment_column, outcome_column, covariate_columns)
        self.truth = 1

    @classmethod
    def simulate_dataset(cls, n_samples, beta):
        W = np.random.uniform(low=-1, high=1, size=(n_samples, 1))
        probs = 1 / (1 + np.exp(-beta * W))
        U = np.random.binomial(1, probs)
        Y = U + np.random.normal(size=(n_samples, 1))
        data = np.concat((W, U, Y), axis=1)
        return cls(data=data, treatment_column=1, outcome_column=2, covariate_columns=[0])


class KangSchaferDataset(ATEDataset):
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        super().__init__(data, treatment_column, outcome_column, covariate_columns)
        self.truth = 210

    @classmethod
    def simulate_dataset(cls, n_samples, c=1):
        xi1 = np.random.normal(size=(n_samples, 1))
        xi2 = np.random.normal(size=(n_samples, 1))
        xi3 = np.random.normal(size=(n_samples, 1))
        xi4 = np.random.normal(size=(n_samples, 1))
        Y = 210 + 27.4 * xi1 + 13.7 * xi2 + 13.7 * xi3 + 13.7 * xi4 + np.random.normal(size=(n_samples, 1))
        logit = -xi1 + 0.5 * xi2 - 0.25 * xi3 - 0.1 * xi4
        logit *= c
        probs = np.exp(logit) / (1 + np.exp(logit))
        U = np.random.binomial(1, probs)
        W1 = np.exp(xi1 / 2)
        W2 = xi2 / (1 + np.exp(xi1)) + 10
        W3 = (xi1 * xi3 / 25 + 0.6) ** 3
        W4 = (xi2 + xi4 + 20) ** 2
        data = np.concat((U, U * Y, W1, W2, W3, W4), axis=1)
        return cls(data=data, treatment_column=0, outcome_column=1, covariate_columns=[2, 3, 4, 5])


class ASEDataset(Dataset):
    def __init__(self, data, treatment_column, outcome_column, covariate_columns, truth=None, folds=None):
        super().__init__(data, treatment_column, outcome_column, covariate_columns, truth, folds)


class ASEExperimentDataset(ASEDataset):
    def __init__(self, data, treatment_column, outcome_column, covariate_columns):
        super().__init__(data, treatment_column, outcome_column, covariate_columns)
        self.truth = 4.670743316303371

    @classmethod
    def simulate_dataset(cls, n_samples):
        W1 = np.random.uniform(low=-3, high=3, size=(n_samples, 1))
        W2 = np.random.uniform(low=-3, high=3, size=(n_samples, 1))
        W3 = np.random.uniform(low=-3, high=3, size=(n_samples, 1))
        W4 = np.random.uniform(low=-3, high=3, size=(n_samples, 1))

        U = np.sin(W2) + W3 + (np.abs(W1) + 0.15) * np.random.exponential(size=(n_samples, 1))
        Y = cls._regression_function(U, W3, W4) + np.random.normal(size=(n_samples, 1))
        data = np.concat((U, Y, W1, W2, W3, W4), axis=1)

        return cls(data=data, treatment_column=0, outcome_column=1, covariate_columns=[2, 3, 4, 5])

    @staticmethod
    def _regression_function(U, W3, W4):
        return U * (W3**2 + np.exp(W4) / 2)


class APEDataset(Dataset):
    def __init__(self, data, outcome_column, covariate_columns, truth=None, folds=None):
        super().__init__(data, None, outcome_column, covariate_columns, truth, folds)


class APEExperimentDataset(APEDataset):
    def __init__(self, data, outcome_column, covariate_columns):
        super().__init__(data, outcome_column, covariate_columns)
        self.truth = 0

    @classmethod
    def simulate_dataset(cls, n_samples):
        X = np.random.uniform(low=-1, high=1, size=(n_samples, 3))
        Y = (X**2).sum(axis=1).reshape(-1, 1) + np.random.normal(size=(n_samples, 1))
        data = np.concat((Y, X), axis=1)
        return cls(data=data, outcome_column=0, covariate_columns=[1, 2, 3])
