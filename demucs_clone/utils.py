import torch.nn as nn


def init_conv_weight_with_rescaling(module):
    if type(module) in [nn.Conv1d, nn.ConvTranspose1d]:
        nn.init.kaiming_uniform_(module.weight)

        alpha = module.weight.data.std().div(0.1)
        module.weight.data.div_(alpha.sqrt())
