import torch
import torch.nn as nn
import torch.nn.functional as func

import models.fusion as fusion

from models.nlp.bert import Bert
from models.optimizer import JointOptimizer
from models.position.four_pos import FourPos, NaiveFourPos
import models.vision.vision_model as vision_model


class Baseline(nn.Module):
    def __init__(self, config):
        super(Baseline, self).__init__()

        self.config = config

        if 'nlp' in config.model.feature:
            self.nlp = Bert(config)

        if 'position' in config.model.feature:
            self.position = NaiveFourPos(config)

        matrix_dim = self.cal_matrix_dim()
        self.linear = nn.Sequential(nn.Linear(matrix_dim*2, 1))


    def cal_matrix_dim(self):
        return self.nlp.output_dim + self.position.output_dim

    def get_feature(self, batch, mode):
        if mode == 'nlp':
            return self.nlp(batch)
        elif mode == 'position':
            return self.position(batch)
        elif mode == 'vision':
            return self.vision(batch)
        else:
            raise NotImplementedError

    def forward(self, batch):
        # note: combine CV and NLP feature together, and use the true edges (cv edges) as metrics
        edge = batch.cv_relation

        feats = {}
        for f in self.config.model.feature:
            feats[f] = self.get_feature(batch, mode=f)

        sent_feat = torch.cat([feats['nlp'], feats['position']], dim=-1)

        sent_lookup = func.embedding(edge, sent_feat)

        neighbors = sent_lookup.narrow(1, 1, sent_lookup.size(1) - 1)
        center = sent_lookup.narrow(1, 0, 1)
        pair_score = self.pair_score(center, neighbors)
        all_score = self.all_score(sent_feat)

        return {'pair_score': pair_score, 'all_score': all_score, 'sent_embedding': None}

    def pair_score(self, x, y):
        x = torch.cat([x for _ in range(y.size(1))], dim=1)
        return self.linear(torch.cat([x, y], dim=-1)).squeeze(-1)

    def all_score(self, embeddings):
        mat = torch.zeros(size=(len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            t = torch.stack([embeddings[i] for _ in range(embeddings.size(0))])
            mat[i] = self.linear(torch.cat([t, embeddings], dim=-1)).squeeze(-1)
        return mat

    def get_optimizer(self, scheduler):
        adam = torch.optim.Adam(self.parameters(),
                                lr=self.config.run.learning_rate, weight_decay=1e-6)
        joint_opt = JointOptimizer(scheduler, adam,
                                   step_size=self.config.run.learning_rate_decay_step,
                                   gamma=self.config.run.learning_rate_decay)

        return joint_opt


