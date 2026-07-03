import torch
from torch import nn

from models.neural_nets.neural_net import NeuralNetwork
from models.neural_nets.utils import ModelConfig


class MADNet(NeuralNetwork):
    def __init__(
        self,
        moment_functional,
        n_covariates,
        model_config: ModelConfig,
        beta=None,
    ):
        super().__init__(moment_functional)
        if beta is None:
            self._beta = lambda covariates, treatment: treatment
        else:
            self._beta = beta
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
        self.beta_perp_branch = nn.Linear(model_config.shared_hidden_layers[-1], 1)

    def _beta_perp_forward(self, covariates, treatment):
        if treatment is not None:
            return self.beta_perp_branch(self.shared_trunk(torch.cat((covariates, treatment), dim=1)))
        else:
            return self.beta_perp_branch(self.shared_trunk(covariates))

    def _beta_minus_beta_perp_forward(self, covariates, treatment):
        return self._beta(covariates, treatment) - self._beta_perp_forward(covariates, treatment)

    def _riesz_forward(self, covariates, treatment):
        beta_minus_beta_perp = self._beta_minus_beta_perp_forward(covariates, treatment)
        denominator = (beta_minus_beta_perp**2).mean().clamp_min(torch.finfo(beta_minus_beta_perp.dtype).eps)
        scale = self.moment_functional(self._beta_minus_beta_perp_forward, covariates, treatment).mean() / denominator
        return scale * beta_minus_beta_perp

    def _outcome_forward(self, covariates, treatment):
        if treatment is not None:
            return self.outcome_branch(self.shared_trunk(torch.cat((covariates, treatment), dim=1)), treatment)
        else:
            return self.outcome_branch(self.shared_trunk(covariates))

    def fit(self, data, training_config):
        def loss_fn(batch):
            outcome_loss = self._get_outcome_mse_loss(batch)
            beta_perp_loss = self._get_beta_perp_loss(batch, training_config.mad_net_lambda)
            return beta_perp_loss + training_config.mad_net_rho * outcome_loss

        self._fit(data=data, loss_fn=loss_fn, training_config=training_config)

    def _get_beta_perp_loss(self, batch, mad_net_lambda):
        covariates, treatment, _ = batch
        beta_perp = self._beta_perp_forward(covariates, treatment)
        beta_loss = nn.functional.mse_loss(beta_perp, self._beta(covariates, treatment))
        moment_constraint = self.moment_functional(self._beta_perp_forward, covariates, treatment).mean().abs()
        return beta_loss + mad_net_lambda * moment_constraint
