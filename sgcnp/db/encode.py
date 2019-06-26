import torch
import torch.nn as nn

import numpy as np


def get_z_score_norm_func(mu, theta, device="cpu"):
    def norm_func(x):
        try:
            x = float(x)
        except:
            x = .0
        x = torch.from_numpy(np.asarray([np.sum([x, -mu]) / theta]).astype(np.float32))
        return x.view(1, -1).to(device)

    return norm_func


# digit > 0
def get_embedding_func(embedding_map, digit, device="cpu"):
    layer = nn.Embedding(len(embedding_map) + 1, digit).to(device)

    if digit > 0:
        def embedding_func(x):
            try:
                index = torch.from_numpy(np.asarray(embedding_map[x]))
            except:
                index = torch.from_numpy(np.asarray([0]))
            index = index.to(device)
            embedding = layer(index)
            return embedding.view(1, -1)
        return embedding_func, layer
    else:
        def embedding_func(x):
            index = torch.from_numpy(np.asarray(embedding_map.get(str(x), 0)).astype(np.float32))
            #     index = torch.from_numpy(np.asarray([0]).astype(np.float32))
            index = index.to(device)
            return index.view(1, -1)
        return embedding_func, None


