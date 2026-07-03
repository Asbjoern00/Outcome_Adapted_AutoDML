import numpy as np
import torch

from datasets.datasets import KangSchaferDataset
from experiments.kangshafer.experiment import OVERLAP_FACTOR, select_lambda_lasso
from experiments.utils import ResultStore
from models.neural_nets.dope_net import DOPENeuralNet
from models.neural_nets.functionals import MeanMissingOutcome
from models.neural_nets.riesz_net import RieszNet
from models.neural_nets.separate_nets import SeparateNeuralNets
from models.neural_nets.utils import ModelConfig, TrainingConfig

MODEL_TYPES = [
    "separate_nets",
    "dope_net_lasso_lambda",
    "riesz_net"
]
N_BOOTSTRAP_SAMPLES = 200
N_EST_SAMPLES = [500,1000,1500,2000]


def run_experiment():
    result_store = ResultStore("results/kangshafer_experiment/bootstrap_experiment.csv")
    for seed in range(1000):
        for model_type in MODEL_TYPES:
            for n_est_samples in N_EST_SAMPLES:
                iteration_identifier = {
                    "seed": seed,
                    "model": model_type,
                    "n_est_samples": n_est_samples,
                }
                if result_store.iteration_completed(iteration_identifier):
                    continue
                row = _run_iteration(
                    seed=seed,
                    model_type=model_type,
                    n_est_samples=n_est_samples,
                )
                result_store.save_rows([row], ["seed", "model", "n_est_samples"])
                result_store.commit()


def _run_iteration(seed, model_type, n_est_samples):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = KangSchaferDataset.simulate_dataset(n_samples=2 * n_est_samples, c=OVERLAP_FACTOR)
    original_estimates, selected_lambda_lasso = _get_estimates(
        data=data,
        model_type=model_type,
        seed=seed,
    )

    bootstrap_estimates = []
    for _ in range(N_BOOTSTRAP_SAMPLES):
        bootstrap_data = data.bootstrap()
        estimates, _ = _get_estimates(
            data=bootstrap_data,
            model_type=model_type,
            seed=seed,
            lambda_lasso=selected_lambda_lasso,
        )
        bootstrap_estimates.append(estimates["point_estimate"])

    return {
        "seed": seed,
        "model": model_type,
        "point_estimate": original_estimates["point_estimate"],
        "var_estimate": original_estimates["var_estimate"],
        "bootstrap_standard_error": np.std(bootstrap_estimates, ddof=1),
        "bootstrap_quantile_2_5": np.quantile(bootstrap_estimates, 0.025),
        "bootstrap_quantile_97_5": np.quantile(bootstrap_estimates, 0.975),
        "n_bootstrap_samples": N_BOOTSTRAP_SAMPLES,
        "lambda_lasso": selected_lambda_lasso,
        "truth": data.truth,
        "n_est_samples": n_est_samples,
    }


def _get_estimates(data, model_type, seed, lambda_lasso=None):
    data.create_folds(2)
    data_train, data_est = data.get_fit_and_test_folds(test_fold_index=1)
    mean, std = data_train.standardize_covariates()
    data_est.standardize_covariates(mean=mean, std=std)

    functional = MeanMissingOutcome()
    model, selected_lambda_lasso = _fit_model(
        model_type=model_type,
        functional=functional,
        data_train=data_train,
        seed=seed,
        lambda_lasso=lambda_lasso,
    )
    return model.get_estimates(data_est), selected_lambda_lasso


def _fit_model(model_type, functional, data_train, seed, lambda_lasso=None):
    model_config = ModelConfig()
    training_config = TrainingConfig()

    if model_type == "separate_nets":
        model = SeparateNeuralNets(
            moment_functional=functional,
            n_covariates=data_train.n_covariates,
            model_config=model_config,
        )
        model.fit_outcome_branch(data=data_train, training_config=training_config)
        model.fit_riesz_branch(data=data_train, training_config=training_config)
    elif model_type == "dope_net_lasso_lambda":
        model = DOPENeuralNet(
            moment_functional=functional,
            n_covariates=data_train.n_covariates,
            model_config=model_config,
        )
        if lambda_lasso is None:
            lambda_lasso = select_lambda_lasso(data=data_train, model=model, seed=seed)
        model.fit_outcome_branch(
            data=data_train,
            training_config=TrainingConfig(lasso_lambda=lambda_lasso),
            pretrain_config=TrainingConfig(patience=3),
        )
        model.freeze_shared_trunk()
        model.fit_riesz_branch(data=data_train, training_config=training_config)
    elif model_type == "riesz_net":
        model = RieszNet(
            moment_functional=functional,
            n_covariates=data_train.n_covariates + 1,
            model_config=model_config,
        )
        model.fit(data=data_train, training_config=training_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, lambda_lasso


if __name__ == "__main__":
    run_experiment()
