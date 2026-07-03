import numpy as np
import torch

from experiments.utils import ResultStore
from models.neural_nets.dope_net import DOPENeuralNet
from datasets.datasets import IHDPDataset
from models.neural_nets.functionals import AverageTreatmentEffect
from models.neural_nets.utils import ModelConfig, TrainingConfig


def run_experiment():
    result_store = ResultStore("results/mae_plot_experiments/ihdp_outcome_adapted.csv")
    for replication_id in range(1, 1001):
        if result_store.iteration_completed({"replication_id": replication_id}):
            continue
        row = _run_iteration(replication_id=replication_id, seed=replication_id)
        result_store.save_rows([row], ["replication_id"])
        result_store.commit()


def _run_iteration(replication_id, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    functional = AverageTreatmentEffect()
    data = IHDPDataset.load_replication(replication_id=replication_id)
    model = DOPENeuralNet(moment_functional=functional, n_covariates=data.n_covariates, model_config=ModelConfig())

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
    }


if __name__ == "__main__":
    run_experiment()
