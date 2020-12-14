import torch
from torch import nn
from torch.nn import functional as func
from transformers import BertConfig, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertPreTrainedModel

from models.optimizer import JointOptimizer, CustomJointOptimizer

LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP = {}

LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LayoutLMConfig(BertConfig):
    pretrained_config_archive_map = LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "bert"

    def __init__(self, max_2d_position_embeddings=1024, **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings


class LayoutLMEmbeddings(nn.Module):
    def __init__(self, config):
        super(LayoutLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids,
            bbox,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        h_position_embeddings = self.h_position_embeddings(
            bbox[:, :, 3] - bbox[:, :, 1]
        )
        w_position_embeddings = self.w_position_embeddings(
            bbox[:, :, 2] - bbox[:, :, 0]
        )
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
                words_embeddings
                + position_embeddings
                + left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
                + h_position_embeddings
                + w_position_embeddings
                + token_type_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutLMModel(BertModel):
    config_class = LayoutLMConfig
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super(LayoutLMModel, self).__init__(config)
        self.embeddings = LayoutLMEmbeddings(config)
        self.init_weights()

    def forward(
            self,
            input_ids,
            bbox,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, bbox, position_ids=position_ids, token_type_ids=token_type_ids
        )
        encoder_outputs = self.encoder(
            embedding_output, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output) + encoder_outputs[
                                                     1:
                                                     ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class LayoutLMForLinkPrediction(BertPreTrainedModel):
    config_class = LayoutLMConfig
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super(LayoutLMForLinkPrediction, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # TODO: how to decide the dim
        matrix_dim = 768
        self.joint_matrix = nn.Parameter(torch.rand(size=(matrix_dim, matrix_dim)), requires_grad=True)

        self.init_weights()

    def forward(
            self,
            input_ids,
            bbox,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            edge=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        #
        # outputs = (logits,) + outputs[
        #                       2:
        #                       ]  # add hidden states and attention if they are here

        sent_feat = pooled_output

        sent_lookup = func.embedding(edge, sent_feat)

        neighbors = sent_lookup.narrow(1, 1, sent_lookup.size(1) - 1)
        center = sent_lookup.narrow(1, 0, 1)
        pair_score = self.pair_score(center, neighbors, self.joint_matrix)
        all_score = self.all_score(sent_feat, self.joint_matrix)

        # fake_label = torch.tensor([0 for _ in range(pair_score.size(0))], dtype=torch.long).to(self.config.run.device)
        # loss = func.cross_entropy(pair_score, fake_label, reduction='mean')

        # outputs = (loss, all_score,) + outputs[2:]
        #
        # return outputs

        # return outputs  # (loss), logits, (hidden_states), (attentions)

        return {'pair_score': pair_score, 'all_score': all_score, 'sent_embedding': None}

    def pair_score(self, x, y, middle):
        middle = torch.stack([middle for _ in range(x.size(0))])
        return (x.bmm(middle)).bmm(y.transpose(1, 2)).squeeze(1)

    def all_score(self, embeddings, middle):
        all_score = embeddings.mm(middle).mm(embeddings.t())
        return all_score


class LayoutLM(nn.Module):
    def __init__(self, config):
        super(LayoutLM, self).__init__()
        self.config = config

        self.bert_config = LayoutLMConfig.from_pretrained(
            config.model.nlp.bert_weight,
            cache_dir=None,
        )

        self.model = LayoutLMForLinkPrediction(self.bert_config)

    def forward(self, batch):
        input_ids = batch.sentence
        attention_mask = batch.mask
        bbox = batch.detailed_position
        edge = batch.neighbor_edge

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "token_type_ids": None,
            "edge": edge
        }

        outputs = self.model(**inputs)

        return outputs

    def get_optimizer(self, scheduler):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.config.run.learning_rate, eps=1e-8
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=self.config.run.total_iter
        )

        joint_opt = CustomJointOptimizer(o=optimizer, s=scheduler)

        return joint_opt

