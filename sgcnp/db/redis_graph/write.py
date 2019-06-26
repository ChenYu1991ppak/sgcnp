import pandas as pd
import hashlib
from tqdm import tqdm
import redis
from collections import Counter
import pymongo

from ...db.redis_graph.attr_map import SchemaCacher
from ...db.redis_graph import RedisBufferSender, NodeComponent, EdgeComponent
from ...config_parser import SchemaInfo

from config import database


NODE_DICT = {}  # global node index dict
TOP_NODE_ID = 0  # current node index
_CNT = None  # node counter
_CNT_IT = None  # node counter iterator


def md5_encode(s):
    md5_obj = hashlib.md5()
    md5_obj.update(s.encode("utf-8"))
    return md5_obj.hexdigest()


class Statistics(object):

    def __init__(self, mongo_url=database):
        self.node_stat = {}
        self.edge_stat = {}

        self.db = pymongo.MongoClient(mongo_url)["gcn_lab"]
        self.col = self.db["graph_stat"]

    def stat_node(self, node_type, num):
        if node_type not in self.node_stat.keys():
            self.node_stat[node_type] = num
        else:
            self.node_stat[node_type] += num

    def stat_edge(self, edge_type, direct, num_list):
        cnt_i = Counter(num_list)
        cnt = {}
        key = direct + "_" + edge_type
        for k, v in cnt_i.items():
            cnt[str(k)] = v

        if key not in self.edge_stat.keys():
            self.edge_stat[key] = cnt
        else:
            for k, v in cnt.items():
                if k not in self.edge_stat[key].keys():
                    self.edge_stat[key][k] = v
                else:
                    self.edge_stat[key][k] += v

    def record(self, graph_name):
        msg = {"node": self.node_stat,
               "edge": self.edge_stat
               }
        # self.col.insert_one({"_id": graph_name, "stat": msg})
        self.col.update({"_id": graph_name}, {"$set":{"_id": graph_name, "stat": msg}}, upsert=True)

        self.node_stat = {}
        self.edge_stat = {}


class GraphWriter(object):

    def __init__(self, redis_host, redis_port, redis_pwd):

        self.schema_dict = None

        self.schema_info = None
        self.schema_id = None
        self.nodes_info = None
        self.edges_info = None

        # GraphDB connection config
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_pwd = redis_pwd

        self.client = None
        self.stat = Statistics()

        self._initialize()

    def _initialize(self):
        self.client = redis.StrictRedis(host=self.redis_host, port=self.redis_port, password=self.redis_pwd)
        self.map_client = redis.StrictRedis(host=self.redis_host, port=self.redis_port, password=self.redis_pwd, db=6)

    def set_schema(self, schema_dict):
        self.schema_dict = schema_dict

        self.schema_info = SchemaInfo(schema_dict)
        self.schema_id = self.schema_info.schema_id
        self.nodes_info = self.schema_info.node_info_list
        self.edges_info = self.schema_info.edge_info_list

    def get_schema(self):
        return self.schema_info

    @staticmethod
    def _derive_encode_func(t, l):
        global _CNT
        global _CNT_IT
        _CNT = tqdm(range(0, l))
        _CNT_IT = iter(_CNT)

        def encode_func(attr_df):
            global _CNT
            joint_str = t
            for key in attr_df.keys():
                joint_str += key + ":" + str(attr_df[key]) + "/"
            _id = md5_encode(joint_str)
            next(_CNT_IT)
            return _id
        return encode_func

    def preprocess(self, data):
        # calculate unique ID of Nodes
        global _CNT
        for node_info in self.nodes_info:
            type_id = node_info.type_id
            node_type = node_info.type
            node_data = pd.DataFrame()
            index_key = None
            for attr_dict in node_info.attrs:
                name = attr_dict["name"]
                source = attr_dict["source"]
                if attr_dict["is_index"] and index_key is None:
                    index_key = name
                node_data[name] = data[source]

            # generate node's id
            print("* gen '%s' ID..." % node_type)
            if index_key is None:
                encode_func = GraphWriter._derive_encode_func(node_type, len(node_data))
                data[type_id] = node_data.apply(encode_func, axis=1)
                _CNT.close()
            else:
                data[type_id] = node_data[index_key]
        return data

    def get_node_component(self, data, node_info):
        global NODE_DICT
        global TOP_NODE_ID
        node_df = pd.DataFrame()
        type_id = node_info.type_id
        node_type = node_info.type
        node_df["id_"] = data[type_id]
        for attr_dict in node_info.attrs:
            attr_name = attr_dict["name"]
            attr_src = attr_dict["source"]
            node_df[attr_name] = data[attr_src]
        # save node type on attributes temporarily
        node_df["ntype"] = node_type

        node_df = node_df.drop_duplicates(subset=["id_"], keep='first')
        for _id in node_df["id_"]:
            if _id not in NODE_DICT:
                NODE_DICT[_id] = TOP_NODE_ID
                TOP_NODE_ID += 1

        cnt = node_df.shape[0]
        self.stat.stat_node(node_type, cnt)

        return NodeComponent(node_df, node_type)

    @staticmethod
    def split_row(df, row_key):
        splited = df[row_key].str.split("/", expand=True).stack().\
            reset_index(level=1, drop=True).rename(row_key)
        return splited, row_key

    @staticmethod
    def split_multi_items(edge_info, data):
        for attr_dict in edge_info.attrs:
            attr_name = attr_dict["name"]
            attr_src = attr_dict["source"]
            seri, key = GraphWriter.split_row(data, attr_src)
            yield seri, attr_name

    def get_edge_component(self, data, edge_info):
        global NODE_DICT
        edge_df = pd.DataFrame()
        edge_type = edge_info.type
        edge_df["src"] = data[edge_info.src_type_id]
        edge_df["tar"] = data[edge_info.tar_type_id]

        multi = edge_info.multi
        splited_seri = []
        if multi is not None:
            seri, _ = GraphWriter.split_row(edge_df, multi)
            edge_df = edge_df.drop(multi, axis=1).join(seri.rename(multi))
            for item in GraphWriter.split_multi_items(edge_info, data):
                splited_seri.append(item)
            for seri, name in splited_seri:
                edge_df.join(seri.rename(name))
        else:
            for attr_dict in edge_info.attrs:
                attr_name = attr_dict["name"]
                attr_src = attr_dict["source"]
                edge_df[attr_name] = data[attr_src]
        edge_df["src"] = edge_df["src"].apply(lambda x: NODE_DICT[x])
        edge_df["tar"] = edge_df["tar"].apply(lambda x: NODE_DICT[x])
        # save edge type on attributes temporarily
        edge_df["etype"] = edge_type
        edge_df = edge_df.drop_duplicates(subset=["src", "tar"], keep='first').reset_index(drop=True)

        edge_num_stat = [f.shape[0] for _, f in edge_df.groupby("src")]
        direct = str(edge_info.src_type_id) + "->" + str(edge_info.tar_type_id)
        self.stat.stat_edge(edge_type, direct, edge_num_stat)

        return EdgeComponent(edge_df, edge_type)

    def df2components(self, data):
        print("preprocessing data...")
        data = self.preprocess(data)
        component_list = []
        print("generating nodes...")
        for node_info in self.nodes_info:
            if not node_info.is_copy:
                comp = self.get_node_component(data, node_info)
                component_list.append(comp)
        print("generating edges...")
        for edge_info in self.edges_info:
            comp = self.get_edge_component(data, edge_info)
            component_list.append(comp)
        return component_list

    def write(self, spark_df, graph_name):
        global NODE_DICT
        global TOP_NODE_ID
        # TODO: msg
        data = spark_df.toPandas()
        # data = spark_df
        components = self.df2components(data)

        # bulk insert to GraphDB
        print("graph data transmitting ...")
        sender = RedisBufferSender(
            client=self.client,
            graph_name=graph_name
        )
        sc = SchemaCacher(self.map_client)
        sc.cache(data, self.schema_dict, graphdataset=graph_name)
        # schema_dict_new = sc.get(graph_name)

        self.stat.record(graph_name)
        num_node, num_rela = sender.insert(components)

        print("Graph: %s has been created %d nodes and %d relations."
              % (graph_name, num_node, num_rela))
        NODE_DICT = {}
        TOP_NODE_ID = 0
        return num_node, num_rela


if __name__ == "__main__":
    pass









