import torch
import torch.nn as nn

from transformers import *


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.config = config
        self.output_dim = config.model.nlp.bert_config.hidden_size
        # self.output_dim = config.model.nlp.lstm * 2

        self.bert_layer = BertModel.from_pretrained(config.model.nlp.bert_weight)
        if config.model.nlp.bert_freeze:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # self.lstm = nn.LSTM(input_size=config.model.nlp.bert_config.hidden_size,
        #                     hidden_size=config.model.nlp.lstm,
        #                     batch_first=True,
        #                     bidirectional=True,
        #                     dropout=0.5)

    def forward(self, batch):
        sent = batch.sentence
        mask = batch.mask

        bert_last_hidden = self.bert_layer(sent, attention_mask=mask)[0]

        bert_cls_hidden = bert_last_hidden[:, 0, :]

        return bert_cls_hidden

        # feat, _ = self.lstm(bert_last_hidden)
        # feat, _ = feat.max(1)

        # _, (hidden, _) = self.lstm(bert_last_hidden)
        # feat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        #
        # return feat
