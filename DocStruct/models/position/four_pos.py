import torch.nn as nn


class FourPos(nn.Module):
    def __init__(self, config):
        super(FourPos, self).__init__()
        self.config = config
        self.output_dim = config.model.position.fc_dim

        self.fc = nn.Sequential(nn.Linear(config.model.position.feature_dim, config.model.position.fc_dim),
                                nn.ReLU(),
                                nn.Linear(config.model.position.fc_dim, config.model.position.fc_dim))

    def forward(self, batch):
        pos = batch.position
        pos_feat = self.fc(pos)
        return pos_feat


class NaiveFourPos(nn.Module):
    def __init__(self, config):
        super(NaiveFourPos, self).__init__()
        self.config = config
        self.output_dim = config.model.position.fc_dim

        self.fc = nn.Sequential(nn.Linear(config.model.position.feature_dim, config.model.position.fc_dim))

    def forward(self, batch):
        pos = batch.position
        pos_feat = self.fc(pos)
        return pos_feat
