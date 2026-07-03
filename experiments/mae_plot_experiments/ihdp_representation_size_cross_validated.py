import numpy as np
import torch

from experiments.utils import ResultStore
from models.neural_nets.dope_net import DOPENeuralNet
from datasets.datasets import IHDPDataset
from models.neural_nets.functionals import AverageTreatmentEffect
from models.neural_nets.utils import ModelConfig, TrainingConfig


def run_experiment():
    result_store = ResultStore("results/mae_plot_experiments/ihdp_representation_size_cross_validated.csv")
    for replication_id in range(1, 1001):
        if result_store.iteration_completed({"replication_id": replication_id}):
            continue
        row = _run_iteration(replication_id=replication_id, seed=replication_id)
        result_store.save_rows([row], ["replication_id"])
        result_store.commit()


def _run_iteration(replication_id, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = IHDPDataset.load_replication(replication_id=replication_id)

    selected_representation_size = select_representation_size(data, seed)

    functional = AverageTreatmentEffect()
    model = DOPENeuralNet(
        moment_functional=functional,
        n_covariates=data.n_covariates,
        model_config=ModelConfig(shared_hidden_layers=[200, 200, selected_representation_size]),
    )
    training_config = TrainingConfig()
    model.fit_outcome_branch(data=data, training_config=training_config)
    model.freeze_shared_trunk()
    model.fit_riesz_branch(data=data, training_config=training_config)

    estimates = model.get_estimates(data)
    truth = data.truth
    return {
        "replication_id": replication_id,
        "seed": seed,
        "point_estimate": estimates["point_estimate"],
        "var_estimate": estimates["var_estimate"],
        "truth": truth,
        "representation_size": selected_representation_size,
    }


def select_representation_size(data, seed):
    functional = AverageTreatmentEffect()
    cv_results = []
    data.create_folds(n_folds=5)
    for representation_size in [1, 3, 10, 200]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = DOPENeuralNet(
            moment_functional=functional,
            n_covariates=data.n_covariates,
            model_config=ModelConfig(shared_hidden_layers=[200, 200, representation_size]),
        )
        cv_result = model.cv_outcome_branch(data=data, training_config=(TrainingConfig()))
        cv_result["representation_size"] = representation_size
        cv_results.append(cv_result)

    best = min(cv_results, key=lambda x: x["cv_loss"])
    threshold = best["cv_loss"] + best["cv_loss_std_error"]
    selected_representation_size = min(
        [cv_result["representation_size"] for cv_result in cv_results if cv_result["cv_loss"] < threshold]
    )
    return selected_representation_size


if __name__ == "__main__":
    run_experiment()
