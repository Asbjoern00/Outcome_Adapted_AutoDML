import numpy as np
import torch

from experiments.utils import ResultStore
from models.neural_nets.riesz_net import RieszNet
from models.neural_nets.functionals import AverageTreatmentEffect
from models.neural_nets.utils import ModelConfig, TrainingConfig
from datasets.datasets import IHDPDataset

RIESZ_WEIGHTS = [1e-2, 1e-1, 1, 10, 100]


def run_experiment():
    result_store = ResultStore("results/varying_weights_plots_experiments/ihdp_varying_riesz_weight.csv")
    for replication_id in range(1, 1001):
        for riesz_weight in RIESZ_WEIGHTS:
            if result_store.iteration_completed({"riesz_weight": riesz_weight, "replication_id": replication_id}):
                continue
            row = _run_iteration(
                replication_id=replication_id,
                seed=replication_id,
                riesz_weight=riesz_weight,
            )
            result_store.save_rows([row], ["riesz_weight", "replication_id"])
            result_store.commit()


def _run_iteration(replication_id, seed, riesz_weight):
    torch.manual_seed(seed)
    np.random.seed(seed)

    functional = AverageTreatmentEffect()
    data = IHDPDataset.load_replication(replication_id)
    truth = data.truth

    model = RieszNet(moment_functional=functional, n_covariates=data.n_covariates + 1, model_config=ModelConfig())
    training_config = TrainingConfig(
        riesz_loss_weights={"riesz": riesz_weight, "outcome": 1, "tmle": 1}, train_proportion=0.6
    )
    model.fit(data, training_config)
    estimates = model.get_estimates(data)
    return {
        "seed": seed,
        "replication_id": replication_id,
        "riesz_weight": riesz_weight,
        "point_estimate": estimates["point_estimate"],
        "var_estimate": estimates["var_estimate"],
        "truth": truth,
    }


if __name__ == "__main__":
    run_experiment()
