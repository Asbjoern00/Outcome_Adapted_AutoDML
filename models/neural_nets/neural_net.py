import copy
import torch
from torch import nn
from abc import ABC, abstractmethod
import math

from models.neural_nets.functionals import MomentFunctional
from models.neural_nets.utils import TrainingConfig, TBranch, SBranch, MLP


class NeuralNetwork(nn.Module, ABC):
    def __init__(self, moment_functional: MomentFunctional):
        super().__init__()
        self.moment_functional = moment_functional

    @abstractmethod
    def _outcome_forward(self, covariates, treatment):
        pass

    @abstractmethod
    def _riesz_forward(self, covariates, treatment):
        pass

    def fit_outcome_branch(self, data, training_config):
        self._fit(data=data, loss_fn=self._get_outcome_mse_loss, training_config=training_config)

    def _get_outcome_mse_loss(self, batch):
        covariates, treatment, outcome = batch
        return nn.functional.mse_loss(self._outcome_forward(covariates, treatment), outcome)

    def fit_riesz_branch(self, data, training_config):
        self._fit(data=data, loss_fn=self._get_riesz_loss, training_config=training_config)

    def _get_riesz_loss(self, batch):
        covariates, treatment, _ = batch
        return (
            self._riesz_forward(covariates, treatment) ** 2
            - 2 * self.moment_functional(self._riesz_forward, covariates, treatment)
        ).mean()

    def _fit(self, data, loss_fn, training_config: TrainingConfig, val_data=None):
        optimizer = self._build_optimizer(training_config)
        scheduler = self._build_scheduler(optimizer, training_config)
        if val_data:
            train_data = data
        else:
            train_data, val_data = data.split_into_train_and_validation_sets(
                train_size=training_config.train_proportion
            )
        train_loader = train_data.create_dataloader(batch_size=training_config.batch_size)
        val_batch = self._move_batch_to_device(val_data.to_batch())
        best = math.inf
        counter = 0
        best_state = copy.deepcopy(self.state_dict())
        for epoch in range(training_config.epochs):
            self.train()
            for batch in train_loader:
                batch = self._move_batch_to_device(batch)
                optimizer.zero_grad()
                loss = loss_fn(batch)
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                val_loss = loss_fn(val_batch)
                scheduler.step(val_loss)
                if val_loss.item() < best - training_config.threshold:
                    best = val_loss.item()
                    counter = 0
                    best_state = copy.deepcopy(self.state_dict())
                else:
                    counter += 1
                    if counter == training_config.patience:
                        self.load_state_dict(best_state)
                        break

    def _build_optimizer(self, training_config: TrainingConfig):
        return torch.optim.Adam(self.parameters(), weight_decay=training_config.weight_decay, lr=training_config.lr)

    def _build_scheduler(self, optimizer, training_config: TrainingConfig):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training_config.scheduler_config.factor,
            patience=training_config.scheduler_config.patience,
            threshold=training_config.scheduler_config.threshold,
            threshold_mode=training_config.scheduler_config.threshold_mode,
            cooldown=training_config.scheduler_config.cooldown,
            min_lr=training_config.scheduler_config.min_lr,
            eps=training_config.scheduler_config.eps,
        )
        return scheduler

    def _move_batch_to_device(self, batch):
        device = next(self.parameters()).device
        if len(batch) == 3:
            covariates, treatment, outcome = batch
            treatment = treatment.to(device)
        else:
            covariates, outcome = batch
            treatment = None
        covariates = covariates.to(device)
        outcome = outcome.to(device)
        return covariates, treatment, outcome

    def get_estimates(self, data, clamp=None):
        covariates, treatment, outcome = self._move_batch_to_device(data.to_batch())
        self.eval()
        with torch.no_grad():
            plugin_terms = self.moment_functional(self._outcome_forward, covariates, treatment)
            riesz_representer = self._riesz_forward(covariates, treatment)
            if clamp:
                riesz_representer = riesz_representer.clamp(-clamp, clamp)
            correction_terms = riesz_representer * (outcome - self._outcome_forward(covariates, treatment))
            dr_terms = plugin_terms + correction_terms

            return {"point_estimate": dr_terms.mean().item(), "var_estimate": dr_terms.var().item()}

    @staticmethod
    def _build_trunk(activation, activation_after_final_shared_layer, dropout_prob, n_covariates, shared_hidden_layers):
        return MLP(
            input_size=n_covariates,
            hidden_sizes=shared_hidden_layers[:-1],
            output_size=(shared_hidden_layers[-1]),
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=activation_after_final_shared_layer,
        )

    @staticmethod
    def _build_branch(activation, dropout_prob, not_shared_hidden_layers, branch_type, representation_size):
        if branch_type == "t_learner":
            return TBranch(
                representation_size=representation_size,
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        elif branch_type == "s_learner":
            return SBranch(
                representation_size=representation_size,
                hidden_sizes=not_shared_hidden_layers,
                activation=activation,
                dropout_prob=dropout_prob,
            )
        elif branch_type == "mlp":
            return MLP(
                input_size=representation_size,
                hidden_sizes=not_shared_hidden_layers,
                output_size=1,
                activation=activation,
                dropout_prob=dropout_prob,
                activation_after_final_layer=False,
            )
        else:
            raise ValueError("Invalid branch type. Must be 't_learner', 's_learner' or 'mlp'.")
