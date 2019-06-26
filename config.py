import os.path

project_path = os.path.abspath('./')

debug = True
address = '0.0.0.0'
port = 80

show_resource = True

database = 'mongodb://localhost:27017'

hdfs_root = project_path
data_root = project_path
data_path = '/data'

temp_path = project_path + '/temp/'
notebook_path = project_path + '/notebook/'

PYTHON = 'python'

SPARK_ENV = ''
SPARK_DIR = 'spark-submit'
SPARK_MASTER = ''
SPARK_CONFIG = {
    "spark.app.name": "gcn-lab",
    "spark.mesos.role": "gcn-lab"
}

redis_host = '127.0.0.1'
redis_port = 6379
redis_pwd = None

spark_file_path = project_path + '/spark'

salt = 'OGhObqhut35pI9bP'

euler_graph_cfg = {
    "graph_dir": "euler_data"
}

try:
    from local_config import *
except:
    pass

IPYTHON_SPARK = SPARK_DIR + ' pyspark-shell-main --name "JupyterNotebookSparkKernel"'
IPYTHON_ENV = {
    'PYSPARK_DRIVER_PYTHON': 'jupyter',
    'PYSPARK_DRIVER_PYTHON_OPTS': 'notebook',
    'PYTHONSTARTUP': spark_file_path + '/shell.py'
}
