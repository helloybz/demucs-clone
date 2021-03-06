# Implementation for Practice: Demucs
[[Original paper]](https://arxiv.org/pdf/1911.13254.pdf) [[Github]](https://github.com/facebookresearch/demucs)
- My own implementation of Demucs in PyTorch.
- Without referencing the original source code.
  (But, referenced for how they have implemented DistributedDataParallel which I had never been used.)

## Dataset
### MUSDB18 [[Link]](https://sigsep.github.io/datasets/musdb.html)
- Following the original paper, MUSDB18 dataset is used to train the model.
- But no extra data is used yet.
- [sigsep-mus-db](https://github.com/sigsep/sigsep-mus-db) package is used to arrange the data.

## Specifications of the model described in the paper
- [ ] 

## TODO
- [x] Epoch Definition: 11-second segments with stride of 1sec, random shift between 0~1, finally 10-second semgent.
- [x] Data augmentation: pitch shift
- [x] Data augmentation: Remixing with the random sources come from the each different tracks.
- [x] Data augmentation: Randomly swapping the channels. (Left, Right)
- [x] Data augmentation: Randomly scaling by a factor between 0.25 and 1.25.
- [x] Data augmentation: Tempo shift
- [x] Quantization: Add regularization term as a proxy for quantization effect
- [x] DistributedDataParallel
- [x] Evaluation: SDR
