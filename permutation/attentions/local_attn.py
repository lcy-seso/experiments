#!/usr/bin/env python3
#coding=utf-8
import pdb
import torch


def sinusoidal_embeddings(seq_len, dim=64, device='cpu'):
    inv_freq = 1. / (10000**(torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.unsqueeze(emb, dim=0)


def rotate_half(x):
    x = x.reshape((x.shape[0], -1, 2, x.shape[-1] // 2))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


if __name__ == '__main__':

    q = torch.randn(8, 2048, 64)
    k = torch.randn(8, 2048, 64)
    v = torch.randn(8, 2048, 64)

    frequences = sinusoidal_embeddings(2048)

    q_pos = q * torch.cos(frequences) + rotate_half(q) * torch.sin(frequences)
    k_pos = q * torch.cos(frequences) + rotate_half(k) * torch.sin(frequences)
