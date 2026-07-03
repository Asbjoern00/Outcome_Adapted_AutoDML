from models.neural_nets.neural_net import NeuralNetwork
from models.neural_nets.utils import ModelConfig


class SeparateNeuralNets(NeuralNetwork):
    def __init__(
        self,
        moment_functional,
        n_covariates,
        model_config: ModelConfig,
    ):
        super().__init__(moment_functional)
        self.outcome_trunk = self._build_trunk(
            activation=model_config.activation,
            activation_after_final_shared_layer=model_config.activation_after_final_shared_layer,
            dropout_prob=model_config.dropout_prob,
            n_covariates=n_covariates,
            shared_hidden_layers=model_config.shared_hidden_layers,
        )
        self.riesz_trunk = self._build_trunk(
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
            return self.outcome_branch(self.outcome_trunk(covariates), treatment)
        else:
            return self.outcome_branch(self.outcome_trunk(covariates))

    def _riesz_forward(self, covariates, treatment):
        if treatment is not None:
            return self.riesz_branch(self.riesz_trunk(covariates), treatment)
        else:
            return self.riesz_branch(self.riesz_trunk(covariates))
