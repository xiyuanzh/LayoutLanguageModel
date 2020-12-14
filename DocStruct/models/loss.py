import math
import torch
import torch.nn as nn
import torch.nn.functional as func


class Loss(nn.Module):
    loss = ['loss']

    def forward(self, feature, batch, train=False):
        raise NotImplementedError


class CrossEntropy(Loss):
    loss = ["loss"]

    def __init__(self, config):
        super(CrossEntropy, self).__init__()
        self.config = config

    def forward(self, feature, batch, train=False):
        # TODO: here still need to solve the batch size problem
        label = batch.label.squeeze(0)
        return {'loss': func.cross_entropy(feature, label, reduction='mean')}


class NegSamplingLoss(Loss):
    loss = ["loss"]

    def __init__(self, config):
        super(NegSamplingLoss, self).__init__()
        self.config = config

    def forward(self, feature, batch, train=False):
        score = feature['pair_score']
        fake_label = torch.tensor([0 for _ in range(score.size(0))], dtype=torch.long).to(self.config.run.device)
        return {'loss': func.cross_entropy(score, fake_label, reduction='mean')}


if __name__ == '__main__':
    # note: here only for test
    pass
    loss = nn.CrossEntropyLoss()
    inputs = torch.randn(1, 5, requires_grad=True)
    target = torch.empty(1, dtype=torch.long).random_(5)
    output = loss(inputs, target)
    print(output)
