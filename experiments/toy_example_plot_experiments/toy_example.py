import numpy as np
import torch

from experiments.utils import ResultStore
from models.neural_nets.dope_net import DOPENeuralNet
from models.neural_nets.separate_nets import SeparateNeuralNets
from models.neural_nets.functionals import AverageTreatmentEffect
from models.neural_nets.utils import ModelConfig, TrainingConfig
from datasets.datasets import ToyExampleDataset

BETAS = [0, 1, 2, 3, 4]
MODEL_TYPES = ["separate_nets", "dope_net", "dope_net_no_dim_reduction"]


def run_experiment():
    result_store = ResultStore("results/toy_example_plot_experiments/toy_example.csv")
    for seed in range(1000):
        for beta in BETAS:
            for model_type in MODEL_TYPES:
                if result_store.iteration_completed({"beta": beta, "seed": seed, "model": model_type}):
                    continue
                row = _run_iteration(seed=seed, beta=beta, model_type=model_type)
                result_store.save_rows([row], ["beta", "seed", "model"])
                result_store.commit()


def _run_iteration(seed, beta, model_type):
    torch.manual_seed(seed)
    np.random.seed(seed)

    functional = AverageTreatmentEffect()
    model_config = ModelConfig()
    data = ToyExampleDataset.simulate_dataset(n_samples=1000, beta=beta)
    truth = data.truth
    data.create_folds(n_folds=2)
    data_fit, data_test = data.get_fit_and_test_folds(test_fold_index=1)

    if model_type == "separate_nets":
        training_config = TrainingConfig()
        model = SeparateNeuralNets(
            moment_functional=functional,
            n_covariates=data.n_covariates,
            model_config=model_config,
        )
        model.fit_outcome_branch(data=data_fit, training_config=training_config)
        model.fit_riesz_branch(data=data_fit, training_config=training_config)
    elif model_type == "dope_net":
        model = DOPENeuralNet(
            moment_functional=functional,
            n_covariates=data.n_covariates,
            model_config=model_config,
        )
        training_config = TrainingConfig(lasso_lambda=1)
        model.fit_outcome_branch(data=data_fit, training_config=training_config)
        model.freeze_shared_trunk()
        model.fit_riesz_branch(data=data_fit, training_config=training_config)
    elif model_type == "dope_net_no_dim_reduction":
        model = DOPENeuralNet(
            moment_functional=functional,
            n_covariates=data.n_covariates,
            model_config=model_config,
        )
        training_config = TrainingConfig()
        model.fit_outcome_branch(data=data_fit, training_config=training_config)
        model.freeze_shared_trunk()
        model.fit_riesz_branch(data=data_fit, training_config=training_config)

    estimates = model.get_estimates(data_test, clamp=100)

    return {
        "seed": seed,
        "beta": beta,
        "point_estimate": estimates["point_estimate"],
        "var_estimate": estimates["var_estimate"],
        "truth": truth,
        "model": model_type,
    }


if __name__ == "__main__":
    run_experiment()
