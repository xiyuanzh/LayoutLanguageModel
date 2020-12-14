import torch
import torch.nn as nn
import torch.nn.functional as func

import models.fusion as fusion

from models.nlp.bert import Bert
from models.optimizer import JointOptimizer
from models.position.four_pos import FourPos
import models.vision.vision_model as vision_model


class JointModel(nn.Module):
    def __init__(self, config):
        super(JointModel, self).__init__()

        self.config = config

        if 'nlp' in config.model.feature:
            self.nlp = Bert(config)

        if 'position' in config.model.feature:
            self.position = FourPos(config)

        if 'vision' in config.model.feature:
            self.vision = getattr(vision_model, config.model.vision.name)(config)

        if config.model.fusion == 'ConcatShiftAddFusion':
            dim = self.nlp.output_dim + self.position.output_dim + self.vision.output_dim
            self.weight_fc = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

        matrix_dim = self.cal_matrix_dim()

        self.joint_matrix = nn.Parameter(torch.rand(size=(matrix_dim, matrix_dim)), requires_grad=True)
        self.fusion = getattr(fusion, config.model.fusion)(config)

    def cal_matrix_dim(self):
        dim = 0
        fusion_method = self.config.model.fusion
        modality = self.config.model.feature
        if fusion_method == 'ConcatFusion':
            if 'nlp' in modality:
                dim += self.nlp.output_dim
            if 'position' in modality:
                dim += self.position.output_dim
            if 'vision' in modality:
                dim += self.vision.output_dim
        elif fusion_method == 'ConcatAddFusion':
            assert len(modality) == 3
            dim = self.nlp.output_dim + self.position.output_dim
        elif fusion_method == 'ConcatShiftAddFusion':
            assert len(modality) == 3
            dim = self.nlp.output_dim + self.position.output_dim

        return dim

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
            # feats.append(func.normalize(self.get_feature(batch, mode=f)))

        if self.config.model.fusion == 'ConcatShiftAddFusion':
            feats['fc'] = self.weight_fc

        sent_feat = self.fusion(feats)

        sent_lookup = func.embedding(edge, sent_feat)

        neighbors = sent_lookup.narrow(1, 1, sent_lookup.size(1) - 1)
        center = sent_lookup.narrow(1, 0, 1)
        pair_score = self.pair_score(center, neighbors, self.joint_matrix)
        all_score = self.all_score(sent_feat, self.joint_matrix)

        return {'pair_score': pair_score, 'all_score': all_score, 'sent_embedding': None}

    def pair_score(self, x, y, middle):
        middle = torch.stack([middle for _ in range(x.size(0))])
        return (x.bmm(middle)).bmm(y.transpose(1, 2)).squeeze(1)

    def all_score(self, embeddings, middle):
        all_score = embeddings.mm(middle).mm(embeddings.t())
        return all_score

    def get_optimizer(self, scheduler):
        adam = torch.optim.Adam(self.parameters(),
                                lr=self.config.run.learning_rate, weight_decay=1e-6)
        joint_opt = JointOptimizer(scheduler, adam,
                                   step_size=self.config.run.learning_rate_decay_step,
                                   gamma=self.config.run.learning_rate_decay)

        return joint_opt


