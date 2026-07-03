import numpy as np
import torch

from datasets.datasets import KangSchaferDataset
from experiments.utils import ResultStore
from models.neural_nets.dope_net import DOPENeuralNet
from models.neural_nets.functionals import MeanMissingOutcome
from models.neural_nets.mad_net import MADNet
from models.neural_nets.riesz_net import RieszNet
from models.neural_nets.separate_nets import SeparateNeuralNets
from models.neural_nets.utils import ModelConfig, TrainingConfig

MODEL_TYPES = [
    "separate_nets",
    "dope_net",
    "dope_net_lasso_lambda",
    "dope_net_representation_size",
    "riesz_net",
    "mad_net",
]

N_EST_SAMPLES = [
    500,
    1000,
    1500,
    2000,
]

OVERLAP_FACTOR = 1.75


def run_experiment():
    result_store = ResultStore("results/kangshafer_experiment/experiment.csv")
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
                row = _run_iteration(seed=seed, model_type=model_type, n_est_samples=n_est_samples)
                result_store.save_rows([row], ["seed", "model", "n_est_samples"])
                result_store.commit()


def _run_iteration(seed, model_type, n_est_samples):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = KangSchaferDataset.simulate_dataset(n_samples=2 * n_est_samples, c=OVERLAP_FACTOR)
    truth = data.truth
    data.create_folds(2)
    data_train, data_est = data.get_fit_and_test_folds(test_fold_index=1)
    mean, std = data_train.standardize_covariates()
    data_est.standardize_covariates(mean=mean, std=std)

    functional = MeanMissingOutcome()
    model = _fit_model(model_type=model_type, functional=functional, data_train=data_train, seed=seed)
    estimates = model.get_estimates(data_est)

    return {
        "seed": seed,
        "model": model_type,
        "point_estimate": estimates["point_estimate"],
        "var_estimate": estimates["var_estimate"],
        "truth": truth,
        "n_est_samples": n_est_samples,
    }


def _fit_model(model_type, functional, data_train, seed):
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
    elif model_type == "dope_net":
        model = DOPENeuralNet(
            moment_functional=functional,
            n_covariates=data_train.n_covariates,
            model_config=model_config,
        )
        model.fit_outcome_branch(data=data_train, training_config=training_config)
        model.freeze_shared_trunk()
        model.fit_riesz_branch(data=data_train, training_config=training_config)
    elif model_type == "dope_net_lasso_lambda":
        model = DOPENeuralNet(
            moment_functional=functional,
            n_covariates=data_train.n_covariates,
            model_config=model_config,
        )
        selected_lambda_lasso = select_lambda_lasso(data=data_train, model=model, seed=seed)
        model.fit_outcome_branch(
            data=data_train,
            training_config=TrainingConfig(lasso_lambda=selected_lambda_lasso),
            pretrain_config=TrainingConfig(patience=3),
        )
        model.freeze_shared_trunk()
        model.fit_riesz_branch(data=data_train, training_config=training_config)
    elif model_type == "dope_net_representation_size":
        selected_representation_size = select_representation_size(
            data=data_train,
            functional=functional,
            seed=seed,
        )
        model = DOPENeuralNet(
            moment_functional=functional,
            n_covariates=data_train.n_covariates,
            model_config=ModelConfig(
                shared_hidden_layers=[200, 200, selected_representation_size],
            ),
        )
        model.fit_outcome_branch(data=data_train, training_config=training_config)
        model.freeze_shared_trunk()
        model.fit_riesz_branch(data=data_train, training_config=training_config)
    elif model_type == "riesz_net":
        model = RieszNet(
            moment_functional=functional,
            n_covariates=data_train.n_covariates + 1,
            model_config=model_config,
        )
        model.fit(data=data_train, training_config=training_config)
    elif model_type == "mad_net":
        model = MADNet(
            moment_functional=functional,
            n_covariates=data_train.n_covariates + 1,
            model_config=model_config,
        )
        model.fit(data=data_train, training_config=training_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def select_lambda_lasso(data, model, seed):
    cv_results = []
    data.create_folds(n_folds=5)
    pretrain_config = TrainingConfig(patience=3)
    for lambda_lasso in [0, 1, 10, 100]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        training_config = TrainingConfig(lasso_lambda=lambda_lasso)
        cv_result = model.cv_outcome_branch(
            data=data,
            training_config=training_config,
            pretrain_config=pretrain_config,
        )
        cv_result["lambda_lasso"] = lambda_lasso
        cv_results.append(cv_result)
    best = min(cv_results, key=lambda x: x["cv_loss"])
    threshold = best["cv_loss"] + best["cv_loss_std_error"]
    return max(cv_result["lambda_lasso"] for cv_result in cv_results if cv_result["cv_loss"] < threshold)


def select_representation_size(data, functional, seed):
    cv_results = []
    data.create_folds(n_folds=5)
    for representation_size in [1, 3, 10, 200]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DOPENeuralNet(
            moment_functional=functional,
            n_covariates=data.n_covariates,
            model_config=ModelConfig(
                shared_hidden_layers=[200, 200, representation_size],
            ),
        )
        cv_result = model.cv_outcome_branch(data=data, training_config=TrainingConfig())
        cv_result["representation_size"] = representation_size
        cv_results.append(cv_result)

    best = min(cv_results, key=lambda x: x["cv_loss"])
    threshold = best["cv_loss"] + best["cv_loss_std_error"]
    return min(cv_result["representation_size"] for cv_result in cv_results if cv_result["cv_loss"] < threshold)


if __name__ == "__main__":
    run_experiment()
