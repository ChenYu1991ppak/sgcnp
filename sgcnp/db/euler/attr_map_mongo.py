from bson.objectid import ObjectId
import pymongo
import numpy as np
from tqdm import tqdm
import pandas as pd
from config import database
import pickle
import os

def process_str(s):
    s = s.replace(".", "/")
    s = s.replace(" ", "#")
    s = s.replace("_", "-")
    return s


class SchemaCacher(object):

    def __init__(self, mongo_url=database):
        self.db = pymongo.MongoClient(mongo_url)["gcn_lab"]
        self.schema_attr_map_dir = "euler_schema"
        if not os.path.exists(self.schema_attr_map_dir):
            os.makedirs(self.schema_attr_map_dir)

    def get_schema_id_by_graph(self, graph_name):
        return self.db["graph2schema"].find_one({"_id": graph_name})["schema_id"]

    def save_graph_schema_map(self, graph_name, schema_id):
        try:
            self.db["graph2schema"].save({"_id": graph_name, "schema_id": schema_id})
        except:
            pass

    def get_schema_dict(self, schema_id):
        return self.db["schema"].find_one({"_id": ObjectId(schema_id)})


    def get_schema_meta(self, schema_dict):
        schema_meta = {"N": {}, "E": {}}

        schema_id = str(schema_dict["_id"])
        for node_dict in schema_dict["nodes"]:
            node_type = node_dict["type"]
            schema_meta["N"][node_type] = {}
            for attr in node_dict["attributes"]:
                schema_meta["N"][node_type][attr["name"]] = \
                    self._get_attr_meta(schema_id, node_type, attr, "N")

        for edge_dict in schema_dict["edges"]:
            edge_type = edge_dict["type"]
            schema_meta["E"][edge_type] = {}
            for attr in edge_dict["attributes"]:
                schema_meta["E"][edge_type][attr["name"]] = \
                    self._get_attr_meta(schema_id, edge_type, attr, "E")
        return schema_meta

    def _get_attr_meta(self, schema_id, type, attr, obj_type):
        '''
        取 schema_node/edge_N/E_attr 的 map_dict
        '''
        assert obj_type == "N" or obj_type == "E"
        map_key = "_".join([process_str(item) for item in [schema_id, type, obj_type, attr["name"]]])
        output_path = self.schema_attr_map_dir + "/" + map_key + ".csv"
        if os.path.exists(output_path):
            map_dict = dict(pd.read_csv(output_path).values)
        else:
            map_dict = {}

        if map_dict == {}:
            if attr["method"] == "index":
                return {"other-": 0}
            else:
                return {}
        else:
            return map_dict


    def _save_attr_meta(self, schema_id, type, attr, obj_type, meta):
        '''
        存 schema_node/edge_N/E_attr 的 map_dict
        '''
        assert obj_type == "N" or obj_type == "E"
        map_key = "_".join([process_str(item) for item in [schema_id, type, obj_type, attr["name"]]])
        output_path = self.schema_attr_map_dir + "/" + map_key + ".csv"
        if os.path.exists(output_path):
            _meta = dict(pd.read_csv(output_path).values)
            _meta.update(meta)
            meta = _meta
        meta_new = {}
        for k, v in meta.items():
            try:
                k = process_str(k)
            except:
                k = "other-"
            meta_new[k] = v
        pd.DataFrame(list(meta_new.items()), columns=['key', 'index']).to_csv(output_path, index=None)

    def _get_attr_meta_0(self, schema_id, type, attr, obj_type):
        '''
        取 schema_node/edge_N/E_attr 的map_dict
        :param schema_id:
        :param type:
        :param attr:
        :param obj_type:
        :return:
        '''
        assert obj_type == "N" or obj_type == "E"
        map_key = "_".join([process_str(item) for item in [schema_id, type, obj_type, attr["name"]]])
        results = list(self.db["attr_map"].find({"_id": {'$regex': map_key}}))
        map_dict = {}

        for r in results:
            key = "_".join(r["_id"].split("_")[4:])
            map_dict[key] = r["value"]
        if map_dict == {}:
            if attr["method"] == "index":
                return {"other-": 0}
            else:
                return {}
        else:
            return map_dict

    def _save_attr_meta_0(self, schema_id, type, attr, obj_type, meta):
        '''
        存 schema_node/edge_N/E_attr 的 map_dict
        '''
        assert obj_type == "N" or obj_type == "E"
        map_key = "_".join([process_str(item) for item in [schema_id, type, obj_type, attr["name"]]])
        meta_new = []
        for k, v in meta.items():
            try:
                k = process_str(k)
            except:
                k = "other-"
            meta_new.append({"_id": "_".join([map_key, k]), "value": v})

        print("saving %s.%s meta" % (type, attr["name"]))
        for m in tqdm(meta_new):
            try:
                self.db["attr_map"].insert(m)
            except:
                pass

    def save_attr_map(self, schema_dict, schema_meta):
        schema_id = str(schema_dict["_id"])
        objs_info = schema_dict["nodes"] + schema_dict["edges"]
        objs_type = ["N" for _ in schema_dict["nodes"]] + ["E" for _ in schema_dict["edges"]]

        for obj_info, obj_type in zip(objs_info, objs_type):
            elem_type = obj_info["type"]
            for attr in obj_info["attributes"]:
                meta = schema_meta[obj_type][elem_type][attr["name"]]
                # try:
                self._save_attr_meta(schema_id, elem_type, attr, obj_type, meta)
                # except:
                #     print("bug: %s  %s" % (elem_type, attr))

    def extract_attr_map(self, schema_id):
        schema_dict = self.get_schema_dict(schema_id)
        schema_meta = self.get_schema_meta(schema_dict)
        return schema_dict, schema_meta

    @staticmethod
    def expand_attr_map(df, schema_dict, schema_meta):
        objs_info = schema_dict["nodes"] + schema_dict["edges"]
        objs_type = ["N" for _ in schema_dict["nodes"]] + ["E" for _ in schema_dict["edges"]]

        schema_meta_new = dict()
        for k in schema_meta.keys():
            schema_meta_new[k] = {}
            for e in schema_meta[k].keys():
                schema_meta_new[k][e] = {}
                for a in schema_meta[k][e].keys():
                    schema_meta_new[k][e][a] = {}

        for obj_info, obj_type in zip(objs_info, objs_type):
            elem_type = obj_info["type"]
            for attr in obj_info["attributes"]:
                attr_meta = schema_meta[obj_type][elem_type][attr["name"]]
                source = attr["source"]
                # index
                if attr["method"] == "index":
                    next_idx = len(attr_meta)
                    for f in df[source]:
                        f = str(f)
                        f = process_str(f)
                        if f not in attr_meta and f not in schema_meta_new[obj_type][elem_type][attr["name"]]:
                            schema_meta_new[obj_type][elem_type][attr["name"]][f] = next_idx
                            next_idx += 1
                # normalize
                elif attr["method"] == "normalize":
                    if attr_meta is None:
                        if np.issubdtype(df[source], np.number):
                            for k, v in df[source].describe().to_dict().items():
                                schema_meta_new[obj_type][elem_type][attr["name"]][str(k)] = float(v)
                        else:
                            # TODO: raise error
                            pass
                    else:
                        pass
                else:
                    # TODO: raise error
                    pass
        return schema_meta_new

    def save_graph_meta(self, graph_name, graph_meta):
        node_dict = graph_meta["node_dict"]
        type_map = graph_meta["type_map"]
        node_attr_dict = graph_meta["node_attr_dict"]
        edge_attr_dict = graph_meta["edge_attr_dict"]
        pd.DataFrame(list(node_dict.items()), columns=['node', 'index']).to_csv("%s/%s_node_dict.csv" % (self.schema_attr_map_dir, graph_name))

        self.db["graph_meta"].save({"_id": graph_name,
                                  "node_dict": {}, "type_map": type_map,
                                  "node_attr_dict": node_attr_dict, "edge_attr_dict": edge_attr_dict})

    def cache(self, df, schema_id, graph_name, graph_meta):
        schema_dict, attr_map = self.extract_attr_map(schema_id)

        attr_map_ex = self.expand_attr_map(df, schema_dict, attr_map)
        self.save_attr_map(schema_dict, attr_map_ex)
        self.save_graph_meta(graph_name, graph_meta)
        self.save_graph_schema_map(graph_name, schema_id)

    def get(self, graph_name):
        schema_id = self.get_schema_id_by_graph(graph_name)
        schema_dict, attr_map = self.extract_attr_map(schema_id)

        for node_cfg in schema_dict["nodes"]:
            node_type = node_cfg["type"]
            for attr in node_cfg["attributes"]:
                if attr["method"] == "index":
                    attr["embedding"] = attr_map["N"][node_type][attr["name"]]
                else:

                    attr["normalize_stat"] = attr_map["N"][node_type][attr["name"]]

        for edge_cfg in schema_dict["edges"]:
            edge_type = edge_cfg["type"]
            for attr in edge_cfg["attributes"]:
                if attr["method"] == "index":
                    attr["embedding"] = attr_map["E"][edge_type][attr["name"]]
                else:
                    attr["normalize_stat"] = attr_map["E"][edge_type][attr["name"]]
        return schema_dict


if __name__ == "__main__":
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
    # sparkconf = SparkConf()
    # for k in spark_cfg.keys():
    #     sparkconf.set(k, spark_cfg[k])
    # spark = SparkSession.builder.config(conf=sparkconf).getOrCreate()
    #
    # parquet_key = ["challenge", "request_time", "duration", "passtime", "new_user",
    #                "drag_count", "from_reg", "x-forwarded-for", "UA",
    #                "ip", "ip_geo", "referer", "captcha_id", "black_flag", "type"]
    #
    # input_path = "hdfs://hadoop-ha/DL/member/zy/label_data/date=18-12-14/hour=12"
    # data_df = spark.read.parquet(input_path).where("type='fullpage'").select(parquet_key).limit(200000)

    schema_id = "5ce50ed10da1b05d5d7174db"

    cacher = SchemaCacher()

    # schema_dict = cacher.get_schema_dict(schema_id)
    # schema_meta = cacher.get_schema_meta(schema_dict)
    # print(schema_meta["N"]["challenge"].keys())

    cacher.get("graph_01")
    # cacher.cache(data_df.toPandas(), schema_id, "test_euler_graph")








