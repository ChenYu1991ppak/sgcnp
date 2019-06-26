import redis
from ..db.redis_graph.attr_map import SchemaCacher

from pyspark import SparkConf
from pyspark.sql import SparkSession

spark_cfg = {
    "spark.app.name": "gcn_platform_new_downloader",
    "spark.mesos.role": "Super-GCN-Platform-test",
    "spark.master": "mesos://zk://10.0.0.1:2181,10.0.0.3:2181,10.0.0.5:2181/mesos",
    "spark.executor.memory": "12G",
    "spark.driver.memory": "12G",
    "spark.cores.max": "8",
    "spark.Kryoserializer.buffer.max": "1280",
    "spark.debug.maxToStringFields": "100",
    "spark.driver.maxResultSize": "25G",
    "spark.executorEnv.PYTHONHASHSEED": "0",
    "spark.dynamicAllocation.enabled": "false",
    "spark.sql.execution.arrow.enabled": "true",
}

sparkconf = SparkConf()
for k in spark_cfg.keys():
    sparkconf.set(k, spark_cfg[k])

spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()

# df = spark.read.parquet("hdfs://hadoop-ha/DL/member/zy/label_data/date=18-12-16/hour=05").limit(1000)
schema_dict = {'_id': "test",
               'edges': [{'attributes': [{'method': 'normalize',
                                          'name': 'passtime',
                                          'source': 'passtime'}],
                          'source': 'node_1',
                          'target': 'node_2',
                          'type': 'Visit'}],
               'nodes': [{'attributes': [{'method': 'index', 'digit': 3, 'name': 'challenge_id', 'source': 'challenge'},
                                         {'method': 'normalize', 'normalize': 'z-score', 'name': 'request_time',
                                          'source': 'request_time'}],
                          'id': 'node_1',
                          'type': 'Challenge'},
                         {'attributes': [{'method': 'index', 'digit': 3, 'name': 'captcha_id', 'source': 'captcha_id'}],
                          'id': 'node_2',
                          'type': 'Captcha'},
                         {'attributes': [{'method': 'index', 'digit': 3, 'name': 'new_user', 'source': 'new_user'},
                                         {'method': 'index', 'digit': 3, 'name': 'UA', 'source': 'UA'}],
                          'id': 'node_3',
                          'type': 'Other'}]}
graphdataset = 'deepknow_12-07'
schemaname = 'test'
client = redis.StrictRedis(host='127.0.0.1', port=6379, password=None, db=6)
sc = SchemaCacher(client)
# 图数据入库
# sc.cache(df, schema_dict, graphdataset=graphdataset)
#
# # 图数据采样
# schema = sc.get(graphdataset)
# print("Normalize stat: ", schema['edges'][0]['attributes'][0]['normalize_stat'])
# print("Index length: ", len(schema['nodes'][0]['attributes'][0]['embedding']))
print("Delete: ", sc._delete(schemaname))

print(sc.get(graphdataset))