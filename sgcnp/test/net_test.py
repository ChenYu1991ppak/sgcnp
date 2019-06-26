from ..model import GCNnet
from ..config_parser import *

import torch
import numpy as np


layers_config = [
    {"idx": "conv_0", "class": "GCNConv", "front": [[None, "input_0"]], "parameter": {"out_channels": 32}},
    {"idx": "bn_1", "class": "BatchNorm1d", "front": [[None, "conv_0"]], "parameter": {}},
    {"idx": "acti_2", "class": "ReLU", "front": [[None, "bn_1"]], "parameter": {}},
    {"idx": "conv_3", "class": "GCNConv", "front": [["add", "acti_2", "conv_0"]], "parameter": {"out_channels": 32}},
    {"idx": "bn_4", "class": "BatchNorm1d", "front": [[None, "conv_3"]], "parameter": {}},
    {"idx": "acti_5", "class": "ReLU", "front": [[None, "bn_4"]], "parameter": {}},
    {"idx": "conv_6", "class": "GCNConv", "front": [[None, "acti_5"]], "parameter": {"out_channels": 32}},
    {"idx": "ln_7", "class": "Linear", "front": [["cat", "conv_6", "conv_3"]], "parameter": {"out_channels": 2}},
    {"idx": "bn_8", "class": "BatchNorm1d", "front": [[None, "ln_7"]], "parameter": {}},
]
inner_cfg = [GCNLayerInfo(lcfg) for lcfg in layers_config]
elems_cfg = {"idx": "gcn_0",
             "elem_type": "net",
             "front": ["dataset_0.edge_index", [None, "dataset_0.x"]],
             "input_dim": [10],
             "inner": inner_cfg,
             "output_idx": ["bn_8", "conv_3"],
             "parameter": {},
             }
elems_info = ElementInfo(elems_cfg)
model = GCNnet(elems_info).to("cuda")

# data
x = np.arange(50).reshape(5, 10)
edge_index = np.asarray([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
x = torch.from_numpy(x).type(torch.FloatTensor).to("cuda")
edge_index = torch.from_numpy(edge_index).type(torch.LongTensor).to("cuda")

# print(model.outputs_dim)
# print(model.inputs_dim)
# print(model.forward(edge_index, x))

param_it = iter(model.parameters())
print(next(param_it))
print(next(param_it))
print(next(param_it))
