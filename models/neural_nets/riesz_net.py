import torch
from torch import nn

from models.neural_nets.neural_net import NeuralNetwork
from models.neural_nets.utils import ModelConfig


class RieszNet(NeuralNetwork):
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
        self.riesz_branch = nn.Linear(model_config.shared_hidden_layers[-1], 1)
        self.epsilon = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def _uncorrected_outcome_forward(self, covariates, treatment):
        if treatment is not None:
            return self.outcome_branch(self.shared_trunk(torch.cat((covariates, treatment), dim=1)), treatment)
        else:
            return self.outcome_branch(self.shared_trunk(covariates))

    def _riesz_forward(self, covariates, treatment):
        if treatment is not None:
            return self.riesz_branch(self.shared_trunk(torch.cat((covariates, treatment), dim=1)))
        else:
            return self.riesz_branch(self.shared_trunk(covariates))

    def _outcome_forward(self, covariates, treatment):
        return self._uncorrected_outcome_forward(covariates, treatment) + self.epsilon * self._riesz_forward(
            covariates, treatment
        )

    def fit(self, data, training_config):
        loss_fn = (
            lambda batch: training_config.riesz_loss_weights["riesz"] * self._get_riesz_loss(batch)
            + training_config.riesz_loss_weights["outcome"] * self._get_uncorrected_outcome_mse_loss(batch)
            + training_config.riesz_loss_weights["tmle"] * self._get_outcome_mse_loss(batch)
        )

        self._fit(data=data, loss_fn=loss_fn, training_config=training_config)

    def _get_uncorrected_outcome_mse_loss(self, batch):
        covariates, treatment, outcome = batch
        outcome_loss = nn.functional.mse_loss(self._uncorrected_outcome_forward(covariates, treatment), outcome)
        return outcome_loss

    def _build_optimizer(self, training_config):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name == "epsilon":
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return torch.optim.Adam(
            [
                {"params": decay_params, "weight_decay": training_config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=training_config.lr,
        )
