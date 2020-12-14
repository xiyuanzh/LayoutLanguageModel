import copy
import json
import random
import logging
from functools import reduce

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.nn.functional import pad

from utils.image_loader import load_image_parts
from utils.linklink_utils import *


class Batch(object):

    def __init__(self, **kwargs):
        """ initialize all input """
        for key, value in kwargs.items():
            self[key] = value

    def __setitem__(self, key, value):
        """ sets the attribute """
        setattr(self, key, value)

    def __getitem__(self, key):
        """ gets the data of the attribute """
        return getattr(self, key, None)

    def to(self, device):
        """ change tensor device """
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                self[key] = self[key].to(device)
        return self


class DocPages(Dataset):
    def __init__(self, pages, config):
        # note: add CV features
        super(DocPages, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(config.model.nlp.bert_weight)

        self.config = config
        self.pages = pages
        self.simple_pages = self.simplify_pages()
        self.input_data = self.prepare_data()

    def get_four_pos(self, pos):
        x0, x1 = min([p[0] for p in pos]), max([p[0] for p in pos])
        y0, y1 = min([p[1] for p in pos]), max([p[1] for p in pos])
        return (x0, y0), (x0, y1), (x1, y0), (x1, y1)

    def simplify_pages(self):
        max_sentence_length = self.config.model.nlp.bert_config.max_position_embeddings

        simple_pages = []
        for page in self.pages.values():
            name = page['name']
            info = page['info']

            # neighbor_relation = page['neighbor_relation']
            # sequence_relation = page['sequence_relation']
            cv_relation = page['cv_relation']

            sentences = []
            positions = []
            detailed_positions = []

            for idx, (key, value) in enumerate(info.items()):

                if self.config.model.name != 'LayoutLM':
                    text = value['text']
                    text_encode_res = self.tokenizer.encode_plus(text, max_length=max_sentence_length, truncation=True)
                    text_encode = text_encode_res['input_ids']
                    sentences.append(text_encode)
                else:
                    detailed_pos_proc = []
                    text_encode = []

                    for word, box in zip(value['text'], value['pos']):

                        box = list(map(lambda x: max(0, min(1000, x)), box))
                        assert 0 <= box[0] <= 1000
                        assert 0 <= box[1] <= 1000
                        assert 0 <= box[2] <= 1000
                        assert 0 <= box[3] <= 1000
                        assert 0 <= box[3] - box[1] <= 1000
                        assert 0 <= box[2] - box[0] <= 1000

                        token = self.tokenizer.tokenize(word)
                        text_encode.extend(token)
                        detailed_pos_proc.extend([box] * len(token))

                    if len(text_encode) > self.tokenizer.max_len_single_sentence:
                        text_encode = text_encode[:self.tokenizer.max_len_single_sentence]
                    text_encode = [self.tokenizer.cls_token] + text_encode + [self.tokenizer.sep_token]
                    text_encode = self.tokenizer.convert_tokens_to_ids(text_encode)

                    cls_box = [0, 0, 0, 0]
                    sep_box = [1, 1, 1, 1]
                    for i in range(4):
                        cls_box[i] = int(1000 * cls_box[i])
                        sep_box[i] = int(1000 * sep_box[i])
                    detailed_pos_proc = [cls_box] + detailed_pos_proc + [sep_box]

                    sentences.append(text_encode)
                    detailed_positions.append(detailed_pos_proc)

                # pos = value['rel_pos'][:4]
                pos = self.get_four_pos(value['rel_pos'])
                pos_flat = []
                for x, y in pos:
                    pos_flat.append(x)
                    pos_flat.append(y)
                positions.append(pos_flat)

            simple_pages.append((name, sentences, positions, cv_relation, detailed_positions))
            # simple_pages.append((name, sentences, positions, neighbor_relation, sequence_relation, detailed_positions))

        return simple_pages

    def get_neg_samples(self, edges, rand_range):

        total_relations = []

        father_dict = {}
        for p, c in edges:
            if c in father_dict:
                father_dict[c].add(p)
            else:
                father_dict[c] = {p}

        for p, c in edges:
            cur = [c, p]

            try_times = 0
            while len(cur) < self.config.dataset.neg_sample + 2:
                rnd = random.randint(0, rand_range - 1)
                try_times += 1
                if rnd not in father_dict[c] or try_times > self.config.dataset.max_try:
                    cur.append(rnd)
                    try_times = 0

            total_relations.append(cur)

        return total_relations

    def prepare_data(self):
        # note: mainly prepare the negative samples
        input_data = []
        for names, sentences, positions, cv_edges, detailed_positions in self.simple_pages:
            cv_total_relations = self.get_neg_samples(cv_edges, len(sentences))
            input_data.append((names, sentences, positions, cv_total_relations, detailed_positions))
        # for names, sentences, positions, neighbor_edges, sequence_edges, detailed_positions in self.simple_pages:
        #     neighbor_total_edges = self.get_neg_samples(neighbor_edges, len(sentences))
        #     sequence_total_edges = self.get_neg_samples(sequence_edges, len(sentences))
        #     input_data.append((names, sentences, neighbor_total_edges, sequence_total_edges, detailed_positions))
        return input_data

    def __getitem__(self, index):
        return self.input_data[index]

    def __len__(self):
        return len(self.input_data)


def collate_fn_wo_image_old(batch):
    names, sentences, positions, cv_total_relations, detailed_positions = zip(*batch)
    names, sentences, positions, cv_total_relations, detailed_positions = list(names), \
                                                                          list(sentences), \
                                                                          list(positions), \
                                                                          list(cv_total_relations), \
                                                                          list(detailed_positions)

    original_sentence_num = [len(sents) for sents in sentences]

    sentences = reduce(lambda x, y: x + y, sentences)
    positions = reduce(lambda x, y: x + y, positions)

    try:
        detailed_positions = reduce(lambda x, y: x + y, detailed_positions)
        detailed_positions = list(map(lambda x: torch.tensor(x, dtype=torch.long), detailed_positions))
        detailed_positions = nn.utils.rnn.pad_sequence(detailed_positions, batch_first=True)
    except:
        pass

    new_cv_relations = copy.deepcopy(cv_total_relations)
    offset = 0
    for batch_id in range(len(new_cv_relations)):
        for edge_id in range(len(new_cv_relations[batch_id])):
            for sent_id in range(len(new_cv_relations[batch_id][edge_id])):
                # note: first add the offset, because we concat different pages
                new_cv_relations[batch_id][edge_id][sent_id] += offset
        offset += original_sentence_num[batch_id]

    sentences = list(map(lambda x: torch.tensor(x, dtype=torch.long), sentences))
    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True)

    positions = torch.tensor(positions, dtype=torch.float)

    new_cv_relations = list(map(lambda x: torch.tensor(x, dtype=torch.long), new_cv_relations))
    new_cv_relations = torch.cat(new_cv_relations)

    attention_masks = (sentences != 0).long()

    return Batch(sentence=sentences, mask=attention_masks, position=positions, image=None,
                 cv_relation=new_cv_relations, detailed_position=detailed_positions)


def collate_fn_wo_image(batch):
    names, sentences, neighbor_total_edges, sequence_total_edges, detailed_positions = map(list, zip(*batch))

    original_sentence_num = [len(sents) for sents in sentences]

    sentences = reduce(lambda x, y: x + y, sentences)
    sentences = list(map(lambda x: torch.tensor(x, dtype=torch.long), sentences))
    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True)

    detailed_positions = reduce(lambda x, y: x + y, detailed_positions)
    detailed_positions = list(map(lambda x: torch.tensor(x, dtype=torch.long), detailed_positions))
    detailed_positions = nn.utils.rnn.pad_sequence(detailed_positions, batch_first=True)

    def _proc_batch_edges(edges):
        new_edges = copy.deepcopy(edges)
        offset = 0
        for batch_id in range(len(new_edges)):
            for edge_id in range(len(new_edges[batch_id])):
                for sent_id in range(len(new_edges[batch_id][edge_id])):
                    # note: first add the offset, because we concat different pages
                    new_edges[batch_id][edge_id][sent_id] += offset
            offset += original_sentence_num[batch_id]
        return new_edges

    new_nb_edges = _proc_batch_edges(neighbor_total_edges)
    new_seq_edges = _proc_batch_edges(sequence_total_edges)

    new_nb_edges = list(map(lambda x: torch.tensor(x, dtype=torch.long), new_nb_edges))
    new_seq_edges = list(map(lambda x: torch.tensor(x, dtype=torch.long), new_seq_edges))

    new_nb_edges = torch.cat(new_nb_edges)
    new_seq_edges = torch.cat(new_seq_edges)

    attention_masks = (sentences != 0).long()

    return Batch(sentence=sentences,
                 mask=attention_masks,
                 sequence_edge=new_seq_edges,
                 neighbor_edge=new_nb_edges,
                 detailed_position=detailed_positions)


def load_data(config):
    logging.info('Loading data begin')

    all_pages = json.load(open(os.path.join(config.dataset.path, "proc.json")))
    split_info = json.load(open(os.path.join(config.dataset.path, "split.json")))

    train_pages = {n: all_pages[n] for n in split_info['train']}
    valid_pages = {n: all_pages[n] for n in split_info['valid']}

    train_dataset = DocPages(train_pages, config)
    val_dataset = DocPages(valid_pages, config)

    train_sampler = DistributedGivenIterationSampler(train_dataset, config.run.total_iter, config.run.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=config.run.batch_size, num_workers=config.run.num_workers,
                              pin_memory=True, sampler=train_sampler, collate_fn=collate_fn_wo_image_old)

    train_val_sampler = DistributedSampler(train_dataset, round_up=False)
    train_val_loader = DataLoader(train_dataset, batch_size=1, num_workers=config.run.num_workers,
                                  pin_memory=True, sampler=train_val_sampler, collate_fn=collate_fn_wo_image_old)
    val_sampler = DistributedSampler(val_dataset, round_up=False)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=config.run.num_workers,
                            pin_memory=True, sampler=val_sampler, collate_fn=collate_fn_wo_image_old)
    return train_loader, train_val_loader, val_loader
