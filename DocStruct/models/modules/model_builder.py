import importlib

EXTERNAL_MODEL_ZOO = {}


def model_builder(config):
    class_num = 1  # block
    with open(config.dataset.dict) as fr:
        for line in fr:
            if len(line.strip()) > 0:
                class_num += 1
    if config.model.arch in EXTERNAL_MODEL_ZOO:
        cls = EXTERNAL_MODEL_ZOO[config.model.arch]
    else:
        module_name, cls_name = config.model.arch.split('.')
        module = importlib.import_module('text_recog.models.backbones.' + module_name)
        cls = getattr(module, cls_name)
    return cls(class_num=class_num, **config.model.normalize, **config.model.kwargs)
