from ..model import GCNloss
from ..config_parser import *

import numpy as np
import torch
import torch.nn.functional as F


loss_cfg = {
    "idx": "rgsloss_0",
    # "front": [[None, "input_0"], [None, "input_1"]],
    # "back": [],
    "class": "MSE",
    "parameter": {}
}
elem2_inner = LossInfo(loss_cfg)
elem2_cfg = {
    "idx": "loss_2",
    "elem_type": "loss",
    "front": [[None, "gcn_1.bn_8"], [None, "dataset_0.y"]],
    "input_dim": [2, 2],
    "inner": elem2_inner,
    "output_idx": [],
    "parameter": {},
}
elem_info = ElementInfo(elem2_cfg)

loss = GCNloss(elem_info)
print(elem2_inner.front)
print(elem2_inner.back)

y_pre = np.asarray([[0., 1]])
y = np.asarray([[0, 1]])
y_idx = np.asarray([1])
y_pre = torch.from_numpy(y_pre).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.FloatTensor)
y_idx = torch.from_numpy(y_idx).type(torch.FloatTensor)
print(loss(y_pre, y))

print(F.mse_loss(y_pre, y))
