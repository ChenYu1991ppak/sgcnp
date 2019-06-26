# -*- coding:utf-8 -*-
'''
web 的shell调用模板 test
'''
import os
from super_gcn_modeling_platform.config_parser import GCNLayerInfo, LossInfo, SourceInfo, ElementInfo, ComputeInfo
from super_gcn_modeling_platform.config_parser.tools import ConfigTransformer
from super_gcn_modeling_platform.db.euler.write import GraphWriter
from super_gcn_modeling_platform.model.compute_graph.engine import ComputeEngine

from super_gcn_modeling_platform.web_api import web_train, CfgWrapper
import pickle

os.chdir(os.path.dirname(os.path.realpath(__file__)))
# config = pickle.load(open('./edge_df2.pkl', 'rb'))

final_config = {'folder': '/www/gcn-lab/instance/instance_1_5ce7af450da1b06d8e0caabf', 'blocks': [], 'graph': [{'id': 'Dataset_1', 'input': {}, 'type': 'Dataset', 'isBlock': True}, {'id': 'GCNConv_1', 'input': ['Dataset_1.x'], 'type': 'GCNConv', 'config': {'out_channels': 20, 'improved': False, 'bias': True}, 'isBlock': False, 'merge': None}, {'id': 'CrossEntropy_1', 'input': {'label': 'Dataset_1.y', 'target': 'GCNConv_1'}, 'type': 'CrossEntropy', 'isBlock': True}], 'dataset': [{'name': 'Dataset_1', 'train': "graph_02", 'test': "graph_02", 'config': {'degree': [10, 10, 10, 10, 10], 'batch_size': 4}}], 'config': {'optimizer': 'SGD', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0001, 'epoch': 2000, 'lr_decay_coef': 0.75, 'lr_decay_gap': 200, 'selected_loss': ['CrossEntropy_1']}}



tr = ConfigTransformer(final_config)
elements_config, _, _ = tr.parse_blocks2elements()
computation_config = tr.parse_graph2computation()
wrapper = CfgWrapper(elements_config, computation_config)
train_info = wrapper.wrap_train_info()

trainer = ComputeEngine(train_info)
trainer.load("model_epoch_10.pkl")
trainer.predict("cpu")

# print(__file__)