import numpy as np
import torch

from experiments.utils import ResultStore
from models.neural_nets.riesz_net import RieszNet
from models.neural_nets.functionals import AverageTreatmentEffect
from models.neural_nets.utils import ModelConfig, TrainingConfig
from datasets.datasets import IHDPDataset

TMLE_WEIGHTS = [0, 0.5, 1, 1.5, 2]


def run_experiment():
    result_store = ResultStore("results/varying_weights_plots_experiments/ihdp_varying_tmle_weight.csv")
    for replication_id in range(1, 1001):
        for tmle_weight in TMLE_WEIGHTS:
            if result_store.iteration_completed({"tmle_weight": tmle_weight, "replication_id": replication_id}):
                continue
            row = _run_iteration(
                replication_id=replication_id,
                seed=replication_id,
                tmle_weight=tmle_weight,
            )
            result_store.save_rows([row], ["tmle_weight", "replication_id"])
            result_store.commit()


def _run_iteration(replication_id, seed, tmle_weight):
    torch.manual_seed(seed)
    np.random.seed(seed)

    functional = AverageTreatmentEffect()
    data = IHDPDataset.load_replication(replication_id)
    truth = data.truth

    model = RieszNet(moment_functional=functional, n_covariates=data.n_covariates + 1, model_config=ModelConfig())
    training_config = TrainingConfig(
        riesz_loss_weights={"riesz": 0.1, "outcome": 2 - tmle_weight, "tmle": tmle_weight}, train_proportion=0.6
    )
    model.fit(data, training_config)
    estimates = model.get_estimates(data)
    return {
        "seed": seed,
        "replication_id": replication_id,
        "tmle_weight": tmle_weight,
        "point_estimate": estimates["point_estimate"],
        "var_estimate": estimates["var_estimate"],
        "truth": truth,
    }


if __name__ == "__main__":
    run_experiment()
