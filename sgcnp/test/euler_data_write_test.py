from pyspark import SparkConf
from pyspark.sql import SparkSession
from bson.objectid import ObjectId

spark_cfg = {
    "spark.app.name": "gcn_platform_new_downloader",
    "spark.mesos.role": "Super-GCN-Platform-test",
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

parquet_key = ["challenge", "request_time", "duration", "passtime", "new_user",
               "drag_count", "from_reg", "x-forwarded-for", "UA",
               "ip", "ip_geo", "referer", "captcha_id", "black_flag", "type"]

schema_dict = {
    "_id": ObjectId("5ce50ed10da1b05d5d7174db"),
    'name': 'test_01',
                    'nodes': [{'type': 'captcha',
                               'attributes': [{'name': 'captcha_id',
                                               'source': 'captcha_id',
                                               'is_index': False,
                                               'method': 'index',
                                               'extract': False,
                                               'digit': '3'}],
                               'id': 'node_0'},
                              {'type': 'challenge',
                               'attributes': [{'name': 'challenge_id',
                                               'source': 'challenge',
                                               'is_index': True,
                                               'method': 'index',
                                               'extract': False,
                                               'digit': '3'},
                                              {'name': 'request_time',
                                               'source': 'request_time',
                                               'is_index': False,
                                               'method': 'index',
                                               'extract': False,
                                               'digit': '3'},
                                              {'name': 'y_',
                                               'source': 'black_flag',
                                               'is_index': False,
                                               'method': 'index',
                                               'extract': False,
                                               'digit': '0'}],
                               'id': 'node_1'},
                              {'type': 'Other',
                               'attributes': [{'name': 'new_user',
                                               'source': 'new_user',
                                               'is_index': False,
                                               'method': 'index',
                                               'extract': False,
                                               'digit': '3'},
                                              {'name': 'UA',
                                               'source': 'UA',
                                               'is_index': False,
                                               'method': 'index',
                                               'extract': False,
                                               'digit': '3'}],
                               'id': 'node_2'}],
                    'edges': [{'type': '',
                               'source': 'node_1',
                               'target': 'node_0',
                               'attributes': [],
                               'multi': None,
                               'directed': False},
                              {'type': '',
                               'source': 'node_1',
                               'target': 'node_2',
                               'attributes': [],
                               'multi': None,
                               'directed': False}],
                    'created_time': '2019-05-22 16:56:49'}

if __name__ == "__main__":
    from ..db.euler.write import GraphWriter

    sparkconf = SparkConf()
    for k in spark_cfg.keys():
        sparkconf.set(k, spark_cfg[k])

    input_path = "hdfs://hadoop-ha/DL/member/zy/label_data/date=18-12-14/hour=12"

    spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()
    data_df = spark.read.parquet(input_path).where("type='fullpage'").select(parquet_key).limit(200000)

    pdf = data_df.toPandas()


    writer = GraphWriter()
    writer.set_schema(schema_dict)
    writer.convert(pdf, "graph_01")

    # print(TYPE_MAP)
    # print(NODE_ATTR_DICT)
    # print(EDGE_ATTR_DICT)
