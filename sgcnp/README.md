# Super_GCN_Modeling_Platform

A Super Powerful Deep Learning Platform for Construction of Graph Model
 
## doc
project overview: https://docs.google.com/document/d/1hymtqDFYwe61_ddiQArU7V1o7xRxc-fEvb4eJTNrNVI/edit#heading=h.ku6vblxfdogo

data protocol   : https://docs.google.com/document/d/11gcX6sdTpEJinm-K9N0f5LmuwNeTP1X97xHp5Yl0CpU/edit#heading=h.8y15zow0bsi8
 
 
## writer graph
from interfaces import web_write

web_write(data, schema, graph_name)

## train
from interfaces import web_train

web_train(web_cfg, device="cuda")


## demo
```bash
git clone git@git.gtapp.xyz:ml/super_gcn_modeling_platform.git
git checkout dev
python3 -m test.euler_data_write_test
python3 -m interfaces
```
