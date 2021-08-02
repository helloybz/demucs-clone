import torch.nn as nn


def init_conv_weight_with_rescaling(module):
    if type(module) in [nn.Conv1d, nn.ConvTranspose1d]:
        nn.init.kaiming_uniform_(module.weight)

        alpha = module.weight.data.std().div(0.1)
        module.weight.data.div_(alpha.sqrt())


def trim_edge(input, target):
    '''
        Trim input tensor as same as target tensors' size
        with aligned on the center of the input.
    '''
    diff = input.shape[-1] - target.shape[-1]
    assert diff >= 0, f'Input should larger or equal than target. {input.shape[-1], target.shape[-1]}'
    input = input[..., diff//2:input.shape[-1]-(diff-diff//2)]

    return input
