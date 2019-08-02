import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()

    return mask


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


if __name__ == "__main__":

    out = get_mask_from_lengths(torch.arange(
        0, 12, out=torch.cuda.LongTensor(12)))
    print(~out)
