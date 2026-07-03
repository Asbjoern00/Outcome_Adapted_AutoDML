import copy
import torch
import numpy as np

from models.neural_nets.neural_net import NeuralNetwork
from models.neural_nets.utils import ModelConfig


class DOPENeuralNet(NeuralNetwork):
    def __init__(
        self,
        moment_functional,
        n_covariates,
        model_config: ModelConfig,
    ):
        super().__init__(moment_functional)
        self.shared_trunk = self._build_trunk(
            activation=model_config.activation,
            activation_after_final_shared_layer=model_config.activation_after_final_shared_layer,
            dropout_prob=model_config.dropout_prob,
            n_covariates=n_covariates,
            shared_hidden_layers=model_config.shared_hidden_layers,
        )
        self.outcome_branch = self._build_branch(
            activation=model_config.activation,
            dropout_prob=model_config.dropout_prob,
            not_shared_hidden_layers=model_config.not_shared_hidden_layers,
            branch_type=model_config.outcome_branch_type,
            representation_size=model_config.shared_hidden_layers[-1],
        )
        self.riesz_branch = self._build_branch(
            activation=model_config.activation,
            dropout_prob=model_config.dropout_prob,
            not_shared_hidden_layers=model_config.not_shared_hidden_layers,
            branch_type=model_config.riesz_branch_type,
            representation_size=model_config.shared_hidden_layers[-1],
        )

    def _outcome_forward(self, covariates, treatment):
        if treatment is not None:
            return self.outcome_branch(self.shared_trunk(covariates), treatment)
        else:
            return self.outcome_branch(self.shared_trunk(covariates))

    def _riesz_forward(self, covariates, treatment):
        if treatment is not None:
            return self.riesz_branch(self.shared_trunk(covariates), treatment)
        else:
            return self.riesz_branch(self.shared_trunk(covariates))

    def fit_outcome_branch(self, data, training_config, pretrain_config=None):
        train_data, val_data = data.split_into_train_and_validation_sets(train_size=training_config.train_proportion)
        if pretrain_config:
            loss_fn = (
                lambda batch: self._get_outcome_mse_loss(batch) + pretrain_config.lasso_lambda * self._lasso_penalty()
            )
            self._fit(data=train_data, val_data=val_data, loss_fn=loss_fn, training_config=pretrain_config)

        loss_fn = lambda batch: self._get_outcome_mse_loss(batch) + training_config.lasso_lambda * self._lasso_penalty()

        self._fit(data=train_data, val_data=val_data, loss_fn=loss_fn, training_config=training_config)

    def freeze_shared_trunk(self):
        for param in self.shared_trunk.parameters():
            param.requires_grad = False

    def _lasso_penalty(self):
        if hasattr(self.shared_trunk.layers[-1], "weight"):
            final_layer_weights = self.shared_trunk.layers[-1].weight
        else:
            final_layer_weights = self.shared_trunk.layers[-2].weight
        return torch.norm(final_layer_weights, dim=1).sum()

    def cv_outcome_branch(self, data, training_config, pretrain_config=None):
        initial_state = copy.deepcopy(self.state_dict())
        cv_results = []
        for i in range(len(data.folds)):
            fit_fold, test_fold = data.get_fit_and_test_folds(test_fold_index=i)
            self.fit_outcome_branch(data=fit_fold, training_config=training_config, pretrain_config=pretrain_config)
            self.eval()
            with torch.no_grad():
                test_batch = self._move_batch_to_device(test_fold.to_batch())
                cv_results.append(self._get_outcome_mse_loss(test_batch).item())
            self.load_state_dict(initial_state)

        return {
            "cv_loss": np.mean(cv_results),
            "cv_loss_std_error": np.std(cv_results, ddof=1) / np.sqrt(len(cv_results)),
        }
