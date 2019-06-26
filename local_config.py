port = 8888
database = 'mongodb://10.0.0.50,10.0.0.51,10.0.0.52/?replicaSet=database'

# PYTHON = 'SPARK_HOME=/opt/spark-2.4.0-bin-hadoop2.6/ PYSPARK_DRIVER_PYTHON=python3.5 PYSPARK_PYTHON=python3.5 python3.5'
# PYTHON = 'python3.7'
PYTHON = 'PYSPARK_DRIVER_PYTHON=python3.5 PYSPARK_PYTHON=python3.5 python3.5'

SPARK_CONFIG = {
    "spark.app.name": "gcn-lab task",
    "spark.mesos.role": "gcn-lab",
    "spark.master": "mesos://zk://10.0.0.1:2181,10.0.0.3:2181,10.0.0.5:2181/mesos",
    "spark.executor.memory": "24G",
    "spark.driver.memory": "24G",
    "spark.cores.max": "24",
    "spark.kryoserializer.buffer.max": "1G",
    "spark.kryoserializer.buffer": "1G",
    "spark.debug.maxToStringFields": "100",
    "spark.driver.maxResultSize": "25G",
    "spark.executorEnv.PYTHONHASHSEED": "0",
    "spark.dynamicAllocation.enabled": "false",
    "spark.sql.execution.arrow.enabled": "true",
}

hdfs_root = 'hdfs://hadoop-ha/'
data_root = '/data/hdfs'
hdfs_url = '/DL/public_data/log_keys_parquet/parquets_product'

salt = 'VKXw0CUUuW'
