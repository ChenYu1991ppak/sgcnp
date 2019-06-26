import os


TRAIN_ROOT_DIR = os.getcwd()



# GraphDB config
redis_graph_cfg = {
    "host": '127.0.0.1',
    "port": 6379,
    "pwd": None,
}

# Euler config
euler_graph_cfg = {
    "graph_dir": "euler_data"
}


# Mongo address
mongo_adr = 'mongodb://10.0.0.50,10.0.0.51,10.0.0.52/?replicaSet=database'








