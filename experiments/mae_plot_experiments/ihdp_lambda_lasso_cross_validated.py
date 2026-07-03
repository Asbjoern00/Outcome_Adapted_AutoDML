import numpy as np
import torch

from experiments.utils import ResultStore
from models.neural_nets.dope_net import DOPENeuralNet
from datasets.datasets import IHDPDataset
from models.neural_nets.functionals import AverageTreatmentEffect
from models.neural_nets.utils import ModelConfig, TrainingConfig


def run_experiment():
    result_store = ResultStore("results/mae_plot_experiments/ihdp_lambda_lasso_cross_validated.csv")
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
    model = DOPENeuralNet(
        moment_functional=functional,
        n_covariates=data.n_covariates,
        model_config=ModelConfig(),
    )

    selected_lambda_lasso = select_lambda_lasso(data, model, seed)

    pretrain_config = TrainingConfig(patience=3)
    training_config = TrainingConfig(lasso_lambda=selected_lambda_lasso)
    model.fit_outcome_branch(data=data, training_config=training_config, pretrain_config=pretrain_config)
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
        "lambda_lasso": selected_lambda_lasso,
    }


def select_lambda_lasso(data, model, seed):
    cv_results = []
    data.create_folds(n_folds=5)
    pretrain_config = TrainingConfig(patience=3)
    for lambda_lasso in [0, 1, 10, 100]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        training_config = TrainingConfig(lasso_lambda=lambda_lasso)
        cv_result = model.cv_outcome_branch(data=data, training_config=training_config, pretrain_config=pretrain_config)
        cv_result["lambda_lasso"] = lambda_lasso
        cv_results.append(cv_result)
    best = min(cv_results, key=lambda x: x["cv_loss"])
    threshold = best["cv_loss"] + best["cv_loss_std_error"]
    selected_lambda_lasso = max(
        [cv_result["lambda_lasso"] for cv_result in cv_results if cv_result["cv_loss"] < threshold]
    )
    return selected_lambda_lasso


if __name__ == "__main__":
    run_experiment()
