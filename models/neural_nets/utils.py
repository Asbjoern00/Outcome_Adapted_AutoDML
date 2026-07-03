import torch
from torch import nn
import dataclasses


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation, dropout_prob, activation_after_final_layer):
        super().__init__()
        layers = []
        input_sizes = [input_size] + hidden_sizes
        output_sizes = hidden_sizes + [output_size]
        for i, (in_size, out_size) in enumerate(zip(input_sizes, output_sizes)):
            layers.append(nn.Linear(in_size, out_size))
            if i + 1 < len(input_sizes) or activation_after_final_layer:
                layers.append(activation())
                if dropout_prob > 0:
                    layers.append(nn.Dropout(dropout_prob))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TBranch(nn.Module):
    def __init__(self, representation_size, hidden_sizes, activation, dropout_prob):
        super().__init__()
        self.t_layers = MLP(
            input_size=representation_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=False,
        )
        self.c_layers = MLP(
            input_size=representation_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=False,
        )

    def forward(self, representation, treatment):
        return self.t_layers(representation) * treatment + self.c_layers(representation) * (1 - treatment)


class SBranch(nn.Module):
    def __init__(self, representation_size, hidden_sizes, activation, dropout_prob):
        super().__init__()
        self.layers = MLP(
            input_size=representation_size + 1,
            hidden_sizes=hidden_sizes,
            output_size=1,
            activation=activation,
            dropout_prob=dropout_prob,
            activation_after_final_layer=False,
        )

    def forward(self, representation, treatment):
        x = torch.cat([representation, treatment], dim=1)
        return self.layers(x)


@dataclasses.dataclass
class ModelConfig:
    shared_hidden_layers: list[int] = dataclasses.field(default_factory=lambda: [200, 200, 200])
    not_shared_hidden_layers: list[int] = dataclasses.field(default_factory=lambda: [100, 100])
    activation: type[nn.Module] = nn.ELU
    dropout_prob: float = 0
    outcome_branch_type: str = "t_learner"
    riesz_branch_type: str = "s_learner"
    activation_after_final_shared_layer: bool = True


@dataclasses.dataclass
class SchedulerConfig:
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5
    threshold: float = 1e-3
    threshold_mode: str = "abs"
    cooldown: int = 0
    min_lr: float = 1e-6
    eps: float = 1e-08


@dataclasses.dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 1000
    lr: float = 1e-3
    patience: int = 30
    weight_decay: float = 1e-3
    train_proportion: float = 0.8
    threshold: float = 1e-3
    lasso_lambda: float = 0
    mad_net_lambda: float = 5
    mad_net_rho: float = 1.0
    scheduler_config: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    riesz_loss_weights: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"riesz": 0.1, "outcome": 1.0, "tmle": 1.0}
    )
