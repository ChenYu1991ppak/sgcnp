# -*- coding:utf-8 -*-
'''
web 的shell调用模板 train
'''
import os

from .web_api import web_train
import pickle

os.chdir(os.path.dirname(os.path.realpath(__file__)))
config = pickle.load(open('./config.pkl', 'rb'))

print(config)

web_train(config)
