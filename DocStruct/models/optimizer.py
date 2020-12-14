import torch


class JointOptimizer:

    def __init__(self, scheduler, *args, **kwargs):
        self.optimizers = []
        for o in args:
            self.optimizers.append(o)

        # note: init scheduler
        if scheduler:
            self.schedulers = []
            for o in args:
                self.schedulers.append(torch.optim.lr_scheduler.StepLR(o,
                                                                       step_size=kwargs['step_size'],
                                                                       gamma=kwargs['gamma']))

    def add_optimizer(self, *args):
        for o in args:
            self.optimizers.append(o)

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()

    def step(self, **kwargs):
        for o in self.optimizers:
            o.step()

        if len(self.schedulers) != 0:
            assert 'epoch' in kwargs
        for s in self.schedulers:
            s.step(kwargs['epoch'])


class CustomJointOptimizer(JointOptimizer):
    def __init__(self, o, s):
        super(CustomJointOptimizer, self).__init__(False)
        self.optimizers = [o]
        self.schedulers = [s]
