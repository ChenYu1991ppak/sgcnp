from ..model.compute_graph.model import ModelProxy
from ..model.compute_graph.engine import ComputeEngine
from ..config_parser import *

import numpy as np
import torch
import torch.nn.functional as F


layers_config = [
    {"idx": "conv_1", "class": "GCNConv", "front": [[None, "input_0"]], "parameter": {"out_channels": 9}},
    {"idx": "bn_2", "class": "BatchNorm1d", "front": [[None, "conv_1"]], "parameter": {}},
    {"idx": "acti_3", "class": "ReLU", "front": [[None, "bn_2"]], "parameter": {}},
    {"idx": "conv_4", "class": "GCNConv", "front": [[None, "acti_3"]], "parameter": {"out_channels": 10}},
    {"idx": "bn_5", "class": "BatchNorm1d", "front": [[None, "conv_4"]], "parameter": {}},
    {"idx": "acti_6", "class": "ReLU", "front": [[None, "bn_5"]], "parameter": {}},
    {"idx": "conv_7", "class": "GCNConv", "front": [[None, "acti_6"]], "parameter": {"out_channels": 11}},
    {"idx": "bn_8", "class": "BatchNorm1d", "front": [[None, "conv_7"]], "parameter": {}},
    {"idx": "acti_9", "class": "ReLU", "front": [[None, "bn_8"]], "parameter": {}},
    {"idx": "conv_10", "class": "GCNConv", "front": [[None, "acti_9"]], "parameter": {"out_channels": 12}},
    {"idx": "bn_11", "class": "BatchNorm1d", "front": [[None, "conv_10"]], "parameter": {}},
    {"idx": "acti_12", "class": "ReLU", "front": [[None, "bn_11"]], "parameter": {}},
    {"idx": "conv_13", "class": "GCNConv", "front": [[None, "acti_12"]], "parameter": {"out_channels": 13}},
    {"idx": "bn_14", "class": "BatchNorm1d", "front": [[None, "conv_13"]], "parameter": {}},
    {"idx": "acti_15", "class": "ReLU", "front": [[None, "bn_14"]], "parameter": {}},
    {"idx": "conv_16", "class": "GCNConv", "front": [[None, "acti_15"]], "parameter": {"out_channels": 14}},
    {"idx": "bn_17", "class": "BatchNorm1d", "front": [[None, "conv_16"]], "parameter": {}},
    {"idx": "acti_18", "class": "ReLU", "front": [[None, "bn_17"]], "parameter": {}},
    {"idx": "conv_19", "class": "GCNConv", "front": [[None, "acti_18"]], "parameter": {"out_channels": 15}},
    {"idx": "bn_20", "class": "BatchNorm1d", "front": [[None, "conv_19"]], "parameter": {}},
    {"idx": "acti_21", "class": "ReLU", "front": [[None, "bn_20"]], "parameter": {}},
    {"idx": "conv_22", "class": "GCNConv", "front": [[None, "acti_21"]], "parameter": {"out_channels": 3}},
    {"idx": "acti_23", "class": "ReLU", "front": [[None, "conv_22"]], "parameter": {}},
]
loss_config = {
    "idx": "clsloss_0",
    "class": "CrossEntropy",
    "parameter": {}
}
dataset_config = {
    "idx": "inductive_0",
    "class": "base",
    "back": ["edge_index", "x", "edge_attr", "y", "mask"],
    "parameter": {
        "train_graph": "graph_01",
        "test_graph": "graph_01",
        "degree": [10, 10, 10, 10, 10],
        "batch_size": 1,
    },
}

elem0_inner = SourceInfo(dataset_config)
elem1_inner = [GCNLayerInfo(cfg) for cfg in layers_config]
elem2_inner = LossInfo(loss_config)

elem0_cfg = {
    "idx": "dataset_0",
    "elem_type": "source",
    "front": [],
    "inner": elem0_inner,
    "output_idx": ["edge_index", "x", "edge_attr", "y", "mask"],
    "parameter": {
        "updated": "ALL",
        "written": "ALL",
    },
}
elem1_cfg = {
    "idx": "gcn_1",
    "elem_type": "model",
    "front": ["dataset_0.edge_index", [None, "dataset_0.x"], [None, "dataset_0.edge_attr"]],
    "inner": elem1_inner,
    "output_idx": ["acti_23"],
    # "ALL" or list of layer idx
    "parameter": {
        "updated": "ALL",
        "written": "ALL",
    }
}
elem2_cfg = {
    "idx": "loss_2",
    "elem_type": "receiver",
    "front": [[None, "gcn_1.acti_23"], [None, "dataset_0.y"], "dataset_0.mask"],
    "inner": elem2_inner,
    "output_idx": [],
    "parameter": {
    },
}

train_inner = [ElementInfo(cfg) for cfg in [elem0_cfg, elem1_cfg, elem2_cfg]]
train_config = {
    "task_idx": "train_1",
    "train_type": "graph",
    "train_inner": train_inner,
    "optimizer_cfg": {"name": "SGD",
                  "parameter": {"lr": 1e-2,
                                "momentum": 0.9,
                                "weight_decay": 1e-4
                                }
                  },
    "epoch": 2000,
    "lr_decay_coef": 0.75,
    "lr_decay_gap": 200,
    "save_dir": "",
    "selected_loss": ["loss_2"],
}
train_info = ComputeInfo(train_config)
trainer = ComputeEngine(train_info)

trainer.train("cuda")
