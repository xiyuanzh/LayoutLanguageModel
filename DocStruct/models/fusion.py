import torch
import torch.nn as nn
import torch.nn.functional as func


class Fusion(nn.Module):
    def __init__(self, config):
        super(Fusion, self).__init__()
        self.config = config

    def forward(self, fusion_dict: dict):
        raise NotImplementedError


class ConcatFusion(Fusion):
    def forward(self, fusion_dict: dict):
        fusion_list = list(fusion_dict.values())
        return torch.cat(fusion_list, dim=-1)


class ConcatAddFusion(Fusion):
    def forward(self, fusion_dict: dict):
        assert len(fusion_dict) == 3
        fusion_list = [fusion_dict['nlp'], fusion_dict['position']]
        cat_feature = torch.cat(fusion_list, dim=-1)
        assert cat_feature.size()[-1] == fusion_dict['vision'].size()[-1]
        return cat_feature + fusion_dict['vision']


class ConcatShiftAddFusion(Fusion):
    def forward(self, fusion_dict: dict):
        nlp_feature = fusion_dict['nlp']
        position_feature = fusion_dict['position']
        vision_feature = fusion_dict['vision']

        fc_layer = fusion_dict['fc']

        base_feature = torch.cat([nlp_feature, position_feature], dim=-1)

        concat_feature = torch.cat([nlp_feature, position_feature, vision_feature], dim=-1)
        weight = fc_layer(concat_feature)
        weighted_vision_feature = weight * vision_feature

        # scale = torch.norm(base_feature, dim=-1) / torch.norm(weighted_vision_feature, dim=-1)
        # one = torch.ones_like(scale)
        # scale_weight = torch.min(one, scale)
        # scale_weight = scale_weight.unsqueeze(-1)

        final_vision_feature = func.normalize(weighted_vision_feature)

        final_feature = base_feature + final_vision_feature

        return final_feature

