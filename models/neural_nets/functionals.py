import torch
from abc import ABC, abstractmethod


class MomentFunctional(ABC):
    @abstractmethod
    def __call__(self, forward_fn, covariates, treatment):
        pass


class AverageTreatmentEffect(MomentFunctional):
    def __call__(self, forward_fn, covariates, treatment):
        return forward_fn(covariates, torch.ones_like(treatment)) - forward_fn(covariates, torch.zeros_like(treatment))


class MeanMissingOutcome(MomentFunctional):
    def __call__(self, forward_fn, covariates, treatment):
        return forward_fn(covariates, torch.ones_like(treatment))


class AverageShiftEffect(MomentFunctional):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def __call__(self, forward_fn, covariates, treatment):
        return forward_fn(covariates, treatment + self.shift) - forward_fn(covariates, treatment)


class AveragePolicyEffect(MomentFunctional):
    def __init__(self, policy_one, policy_zero, n_samples):
        super().__init__()
        self.policy_one = policy_one
        self.policy_zero = policy_zero
        self.n_samples = n_samples
        self.policy_one_covariates = self.policy_one(self.n_samples)
        self.policy_zero_covariates = self.policy_zero(self.n_samples)

    def __call__(self, forward_fn, covariates, treatment):
        return (forward_fn(self.policy_one_covariates, None) - forward_fn(self.policy_zero_covariates, None)).mean()
