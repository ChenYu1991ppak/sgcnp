from ..db import GraphWriter

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

parquet_key = ["challenge", "request_time", "duration", "passtime", "new_user",
               "drag_count", "from_reg", "x-forwarded-for", "UA",
               "ip", "ip_geo", "referer", "captcha_id", "black_flag", "type"]


# schema配置
schema_dict = {
    "_id": "schema_01",
    ## 节点类型
    "nodes": [
        {
            "id": "node_1",
            "type": "Challenge",
            "attributes": [
                {
                    # 节点属性名
                    "name": "challenge_id",
                    # 属性值来源
                    "source": "challenge",
                    # 属性编码方法
                    "method": "index",
                    # 是否唯是一标识
                    "is_index": True,
                    "digit": 3
                },
                {
                    "name": "request_time",
                    "source": "passtime",
                    "method": "normalize",
                },
                {
                    # 作为训练标记的属性
                    "name": "y_",
                    "source": "black_flag",
                    "method": "index",
                    "digit": 0
                }
            ]
        },
        {
            "id": "node_2",
            "type": "Captcha",
            "attributes": [
                {
                    "name": "captcha_id",
                    "source": "captcha_id",
                    "method": "index",
                    "is_index": True,
                    "digit": 3
                }
            ]
        },
        {
            "id": "node_3",
            "type": "Other",
            "attributes": [
                {
                    "name": "new_user",
                    "source": "new_user",
                    "method": "index",
                    "digit": 3
                },
                {
                    "name": "UA",
                    "source": "UA",
                    "method": "index",
                    "digit": 3
                }
            ]
        }
    ],
    "edges": [
        {
            # 边类型
            "type": "Visit",
            # 源节点类型
            "source": "node_1",
            # 目标节点类型
            "target": "node_2",
            # 一对多，或一对一
            # multi: "src", "tar", or None
            "multi": None,
            # 边属性
            "attributes": [
                {
                    "name": "passtime",
                    "source": "passtime",
                    "method": "normalize"
                },
            ]
        },
        {
            "type": "Visit",
            "source": "node_2",
            "target": "node_1",
            "attributes": [
                {
                    "name": "passtime",
                    "source": "passtime",
                    "method": "normalize"
                },
            ]
        },
        {
            "type": "Attr",
            "source": "node_1",
            "target": "node_3",
            "attributes": []
        },
        {
            "type": "Attr",
            "source": "node_3",
            "target": "node_1",
            "attributes": []
        },
    ],
}



schema_dict2 = {
  "_id": "5ce50ed10da1b05d5d7174db",
  "name": "test_01",
  "nodes": [
    {
      "type": "captcha",
      "attributes": [
        {
          "name": "captcha_id",
          "source": "captcha_id",
          "is_index": False,
          "method": "index",
          "extract": False,
          "digit": "3"
        }
      ],
      "id": "node_0"
    },
    {
      "type": "challenge",
      "attributes": [
        {
          "name": "challenge_id",
          "source": "challenge",
          "is_index": True,
          "method": "index",
          "extract": False,
          "digit": "3"
        },
        {
          "name": "request_time",
          "source": "request_time",
          "is_index": False,
          "method": "index",
          "extract": False,
          "digit": "3"
        },
        {
          "name": "y_",
          "source": "black_flag",
          "is_index": False,
          "method": "normalize",
          "extract": False,
          "normalize": "z-score"
        }
      ],
      "id": "node_1"
    },
    {
      "type": "Other",
      "attributes": [
        {
          "name": "new_user",
          "source": "new_user",
          "is_index": False,
          "method": "index",
          "extract": False,
          "digit": "3"
        },
        {
          "name": "UA",
          "source": "UA",
          "is_index": False,
          "method": "index",
          "extract": False,
          "digit": "3"
        }
      ],
      "id": "node_2"
    }
  ],
  "edges": [
    {
      "type": "",
      "source": "node_1",
      "target": "node_0",
      "attributes": [],
      "multi": None,
      "directed": False
    },
    {
      "type": "",
      "source": "node_1",
      "target": "node_2",
      "attributes": [],
      "multi": None,
      "directed": False
    }
  ],
  "created_time": "2019-05-22 16:56:49"
}


# schema_info = SchemaInfo(schema_dict)
def run():
    sparkconf = SparkConf()
    for k in spark_cfg.keys():
        sparkconf.set(k, spark_cfg[k])

    input_path = "hdfs://hadoop-ha/DL/member/zy/label_data/date=18-12-14/hour=12"

    spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()
    data_df = spark.read.parquet(input_path).where("type='fullpage'").select(parquet_key).limit(200000)

    writer = GraphWriter(redis_host='127.0.0.1', redis_port=6379, redis_pwd=None)
    writer.set_schema(schema_dict)
    writer.write(data_df, "20w_test_3")


if __name__ == "__main__":
    run()
