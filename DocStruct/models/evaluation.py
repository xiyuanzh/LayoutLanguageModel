import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics


# note: for eval class:
#  return a dict for each kind of values
#  some of them are meaningful
#  the last value must be 'eval', and this is always the bigger the better

class Eval(nn.Module):
    result = ['eval']

    def forward(self, feature, batch):
        raise NotImplementedError


class F1(Eval):
    result = ['acc', 'recall', 'f1', 'eval']

    def __init__(self, config):
        super(F1, self).__init__()
        self.config = config
        self.category = config.eval.acc.category

    def forward(self, feature, batch):
        # TODO: here need to solve the batch size problem
        label = np.array(batch.label.squeeze(0).cpu())
        predict = np.array(feature.max(-1)[1].cpu())

        acc = metrics.accuracy_score(y_true=label, y_pred=predict)
        recall = metrics.recall_score(y_true=label, y_pred=predict)
        f1 = metrics.f1_score(y_true=label, y_pred=predict)

        return {'acc': acc,
                'recall': recall,
                'f1': f1,
                'eval': f1}


class Acc(Eval):
    result = ['eval']

    def __init__(self, config):
        super(Acc, self).__init__()
        self.config = config
        self.category = config.eval.acc.category

    def forward(self, feature, batch):
        # TODO: here need to solve the batch size problem
        label = np.array(batch.label.squeeze(0).cpu())
        predict = np.array(feature.max(-1)[1].cpu())

        acc = metrics.accuracy_score(y_true=label, y_pred=predict)

        return {'acc': acc,
                'eval': acc}


class Reconstruct(Eval):
    result = ['mAP', 'mRank', 'eval']

    def __init__(self, config):
        super(Reconstruct, self).__init__()
        self.config = config

        self.eval_key = config.eval.key

    def switch_mode(self, mode):
        # assert mode in ['nlp', 'cv', 'joint']
        # switch_dict = {'nlp': 'nlp_relation',
        #                'cv': 'cv_relation',
        #                'joint': 'cv_relation'}
        self.eval_key = self.config.eval.key

    def forward(self, feature, batch):
        adj = {}

        for edge in batch[self.eval_key]:
            c, p = int(edge[0]), int(edge[1])
            if c in adj:
                adj[c].add(p)
            else:
                adj[c] = {p}

        obj_num = len(batch.sentence)
        objects = list(range(obj_num))

        ranksum = nranks = ap_scores = iters = 0

        labels = np.empty(obj_num)

        for obj in objects:
            labels.fill(0)
            if obj not in adj:
                continue
            neighbors = np.array(list(adj[obj]))
            cur_dists = feature['all_score'][obj]
            cur_dists[obj] = -1e12

            sorted_dists, sorted_idx = cur_dists.sort(descending=True)
            ranks, = np.where(np.in1d(sorted_idx.detach().cpu().numpy(), neighbors))
            # The above gives us the position of the neighbors in sorted order.  We
            # want to count the number of non-neighbors that occur before each neighbor
            ranks += 1
            N = ranks.shape[0]

            # To account for other positive nearer neighbors, we subtract (N*(N+1)/2)
            # As an example, assume the ranks of the neighbors are:
            # 0, 1, 4, 5, 6, 8
            # For each neighbor, we'd like to return the number of non-neighbors
            # that ranked higher than it.  In this case, we'd return 0+0+2+2+2+3=14
            # Another way of thinking about it is to return
            # 0 + 1 + 4 + 5 + 6 + 8 - (0 + 1 + 2 + 3 + 4 + 5)
            # (0 + 1 + 2 + ... + N) == (N * (N + 1) / 2)
            # Note that we include `N` to account for the source embedding itself
            # always being the nearest neighbor
            ranksum += ranks.sum() - (N * (N - 1) / 2)
            nranks += ranks.shape[0]
            labels[neighbors] = 1
            ap_scores += metrics.average_precision_score(labels, cur_dists.detach().cpu().numpy())
            iters += 1

        mRank = float(ranksum) / nranks
        mAP = ap_scores / iters

        return {'mRank': mRank,
                'mAP': mAP,
                'eval': mAP}
    

class TopKAcc(Eval):
    result = ['eval']
    
    def __init__(self, config):
        super(TopKAcc, self).__init__()
        self.config = config
        self.Ks = config.eval.top_k_acc.k
        for k in self.Ks:
            self.result.append(f'top{k}')

        self.eval_key = config.eval.key

    def switch_mode(self, mode):
        # assert mode in ['nlp', 'cv', 'joint']
        # switch_dict = {'nlp': 'nlp_relation',
        #                'cv': 'cv_relation',
        #                'joint': 'cv_relation'}
        self.eval_key = self.config.eval.key

    def forward(self, feature, batch):
        right_answers = {}  # the true father for each node
        for edge in batch[self.eval_key]:
            c, p = int(edge[0]), int(edge[1])
            if c in right_answers:
                right_answers[c].add(p)
            else:
                right_answers[c] = {p}

        predict_score = torch.tensor(feature['all_score'])

        result = {}

        for k in self.Ks:
            _, predict_result = predict_score.topk(k, dim=-1, largest=True, sorted=True)
            predict_result = predict_result.detach().cpu().numpy()

            object_num = len(right_answers)
            right_num = 0

            for c, ps in right_answers.items():
                for x in predict_result[c]:
                    if x in ps:
                        right_num += 1
                        break

            top_k_acc = right_num / object_num

            result[f'top{k}'] = top_k_acc

        result['eval'] = result[f'top{min(self.Ks)}']

        return result


class JointEval(Eval):
    result = ['eval']

    def __init__(self, config):
        super(JointEval, self).__init__()
        self.config = config
        self.evaluators = {n: eval(n)(config) for n in config.eval.joint_eval.sub}
        self.eval_sub = config.eval.joint_eval.eval_sub

        for k, e in self.evaluators.items():
            sub_keys = e.result
            for sub_key in sub_keys:
                if sub_key == 'eval':
                    continue
                self.result.append(f'{sub_key}')

    def forward(self, feature, batch):
        result = {}
        for name, evaluator in self.evaluators.items():
            sub_results = evaluator(feature, batch)
            for sub_name, sub_res in sub_results.items():
                if sub_name == 'eval':
                    if name == self.eval_sub:
                        result['eval'] = sub_res
                else:
                    sub_key = f'{sub_name}'
                    result[sub_key] = sub_res
        assert 'eval' in result
        return result
