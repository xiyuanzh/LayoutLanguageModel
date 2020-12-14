import linklink as link
import numpy as np

from linklink import get_world_size, get_rank


def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(ranks=rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank // group_size]


class GroupSyncBatchNorm(link.nn.SyncBatchNorm2d):
    group_by_size = {}

    def __init__(self,
                 num_features,
                 bn_group_size=None,
                 momentum=0.1,
                 sync_stats=True,
                 var_mode=link.syncbnVarMode_t.L2):

        self.group_size = bn_group_size

        super(GroupSyncBatchNorm, self).__init__(
            num_features,
            momentum=momentum,
            group=self._get_group(bn_group_size),
            sync_stats=sync_stats,
            var_mode=var_mode
        )

    @staticmethod
    def _get_group(bn_group_size):
        if bn_group_size in GroupSyncBatchNorm.group_by_size:
            return GroupSyncBatchNorm.group_by_size[bn_group_size]

        rank = get_rank()
        world_size = get_world_size()
        if bn_group_size is None:
            bn_group_size = world_size
        assert world_size % bn_group_size == 0
        bn_group_comm = simple_group_split(world_size, rank, world_size // bn_group_size)
        GroupSyncBatchNorm.group_by_size[bn_group_size] = bn_group_comm
        return bn_group_comm

    def __repr__(self):
        return ('{name}({num_features},'
                ' eps={eps},'
                ' momentum={momentum},'
                ' affine={affine},'
                ' group={group},'
                ' group_size={group_size},'
                ' sync_stats={sync_stats},'
                ' var_mode={var_mode})'.format(
            name=self.__class__.__name__, **self.__dict__))


def BNFunc(group_size):
    def func(*args, **kwargs):
        return GroupSyncBatchNorm(*args, **kwargs, bn_group_size=group_size, sync_stats=True)

    return func
