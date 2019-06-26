import os
import networkx as nx
import pandas as pd
import hashlib
from tqdm import tqdm
import json
import pymongo
from collections import Counter

from ...db.euler.attr_map_mongo import SchemaCacher
from ...config_parser import SchemaInfo

from config import database


TOP_NODE_ID = 0  # current node index
NODE_DICT = {}  # global node index dict
TYPE_MAP = {}
NODE_ATTR_DICT = {}
EDGE_ATTR_DICT = {}

_CNT = None  # node counter
_CNT_IT = None  # node counter iterator


def process_str(s):
    s = s.replace(".", "/")
    s = s.replace(" ", "#")
    s = s.replace("_", "-")
    return s


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
        self.col.insert_one({"_id": graph_name, "stat": msg})

        self.node_stat = {}
        self.edge_stat = {}


class GraphWriter(object):

    def __init__(self, save_dir="euler_data"):
        self.save_dir = save_dir
        # if not os.path.exists(self.save_dir):
        #     os.mkdir(self.save_dir)

        self.schema_dict = None

        self.schema_info = None
        self.schema_id = None
        self.nodes_info = None
        self.edges_info = None

        self.stat = Statistics()

    def set_schema(self, schema_dict):
        self.schema_dict = schema_dict

        self.schema_info = SchemaInfo(schema_dict)
        self.schema_id = self.schema_info.schema_id
        self.nodes_info = self.schema_info.node_info_list
        self.edges_info = self.schema_info.edge_info_list

    def get_schema(self):
        return self.schema_info

    def _get_schema_meta(self):
        global TYPE_MAP
        global NODE_ATTR_DICT
        global EDGE_ATTR_DICT

        meta = {
            "node_type_num": 0,
            "edge_type_num": 0,
            "node_uint64_feature_num": 1,
            "node_float_feature_num": 0,
            "node_binary_feature_num": 0,
            "edge_uint64_feature_num": 1,
            "edge_float_feature_num": 0,
            "edge_binary_feature_num": 0
        }
        node_id_cnt = 0
        edge_id_cnt = 0

        max_node_binary_len = 0
        max_node_float_len = 0
        for node_info in self.schema_info.node_info_list:
            node_type = node_info.type
            if node_type not in TYPE_MAP.keys():
                TYPE_MAP[node_type] = node_id_cnt
                NODE_ATTR_DICT[str(node_id_cnt)] = {}
                meta["node_type_num"] += 1
                float_cnt = 0
                binary_cnt = 0
                for attr in node_info.attrs:
                    if attr["method"] == "index":
                        NODE_ATTR_DICT[str(node_id_cnt)][attr["name"]] = ("binary", binary_cnt)
                        binary_cnt += 1
                    else:
                        NODE_ATTR_DICT[str(node_id_cnt)][attr["name"]] = ("float", float_cnt)
                        float_cnt += 1
                max_node_binary_len = max(max_node_binary_len, binary_cnt)
                max_node_float_len = max(max_node_float_len, float_cnt)
                node_id_cnt += 1
        meta["node_binary_feature_num"] = max_node_binary_len
        meta["node_float_feature_num"] = max_node_float_len

        max_edge_binary_len = 0
        max_edge_float_len = 0
        for edge_info in self.schema_info.edge_info_list:
            edge_type = edge_info.type
            if edge_type not in TYPE_MAP.keys():
                TYPE_MAP[edge_type] = edge_id_cnt
                EDGE_ATTR_DICT[str(edge_id_cnt)] = {}
                meta["edge_type_num"] += 1
                float_cnt = 0
                binary_cnt = 0
                for attr in edge_info.attrs:
                    if attr["method"] == "index":
                        EDGE_ATTR_DICT[str(edge_id_cnt)][attr["name"]] = ("binary", binary_cnt)
                        binary_cnt += 1
                    else:
                        EDGE_ATTR_DICT[str(edge_id_cnt)][attr["name"]] = ("float", float_cnt)
                        float_cnt += 1
                max_edge_binary_len = max(max_edge_binary_len, binary_cnt)
                max_edge_float_len = max(max_edge_float_len, float_cnt)
                edge_id_cnt += 1
        meta["edge_binary_feature_num"] = max_edge_binary_len
        meta["edge_float_feature_num"] = max_edge_float_len
        return meta

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

    def get_node_df(self, data, node_info):
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
        node_df["id_"] = node_df["id_"].apply(lambda x: NODE_DICT[x])
        return node_df

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

    def get_edge_df(self, data, edge_info):
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
        return edge_df

    def parse_df2nx(self, data):
        def push_node2graph(df, graph):
            keys = df.keys().tolist()
            for values in df.values:
                attr = {}
                for k, v in zip(keys, values):
                    attr[k] = v
                id_ = attr.pop("id_")
                graph.add_node(id_, **attr)
            return graph

        def push_edge2graph(df, graph):
            keys = df.keys().tolist()
            for values in df.values:
                attr = {}
                for k, v in zip(keys, values):
                    attr[k] = v
                src = attr.pop("src")
                tar = attr.pop("tar")
                graph.add_edge(src, tar, **attr)
            return graph

        G = nx.Graph()
        for node_info in self.nodes_info:
            node_df = self.get_node_df(data, node_info)
            G = push_node2graph(node_df, G)

        for edge_info in self.edges_info:
            edge_df = self.get_edge_df(data, edge_info)
            G = push_edge2graph(edge_df, G)
        return G

    def parse_df2euler(self, G):
        global TYPE_MAP

        edge_cnt = 0
        for k in TYPE_MAP.keys():
            if k[:2] == "E_":
                edge_cnt += 1

        for node in G.nodes:
            # get attributes of node
            attr = {}
            for k, v in G.node[node].items():
                attr[k] = v
            # init
            buf = dict()
            buf["node_id"] = node
            node_type = TYPE_MAP[attr.pop("ntype")]
            buf["node_type"] = node_type
            buf["node_weight"] = 1

            # add neighbor node
            buf["neighbor"] = {}
            for i in range(edge_cnt):
                buf["neighbor"][str(i)] = dict()
            for n in G[node]:
                buf["neighbor"][str(TYPE_MAP[str(G[node][n]["etype"])])][str(n)] = 1

            # add node features
            buf["uint64_feature"] = {}
            buf["float_feature"] = {}
            buf["binary_feature"] = {}

            buf["uint64_feature"]["0"] = [node_type]
            attr_meta = NODE_ATTR_DICT[str(node_type)]
            for k, v in attr.items():
                attr_type = attr_meta[k][0]
                attr_ord = attr_meta[k][1]
                if attr_type == "binary":
                    buf["binary_feature"][str(attr_ord)] = process_str(str(v))[:5]
                else:
                    buf["float_feature"][str(attr_ord)] = [float(v)]

            # add edge features
            buf["edge"] = []
            for tar in G[node]:
                eattr = {}
                for k, v in G[node][tar].items():
                    eattr[k] = v
                ebuf = dict()
                ebuf["src_id"] = node
                ebuf["dst_id"] = tar
                edge_type = TYPE_MAP[eattr.pop("etype")]
                ebuf["edge_type"] = edge_type
                ebuf["weight"] = 1
                ebuf["uint64_feature"] = {}
                ebuf["float_feature"] = {}
                ebuf["binary_feature"] = {}

                ebuf["uint64_feature"]["0"] = [edge_type]
                eattr_meta = EDGE_ATTR_DICT[str(edge_type)]
                for k, v in eattr.items():
                    eattr_type = eattr_meta[k][0]
                    eattr_ord = eattr_meta[k][1]
                    if eattr_type == "binary":
                        ebuf["binary_feature"][str(eattr_ord)] = process_str(str(v))[:5]
                    else:
                        ebuf["float_feature"][str(eattr_ord)] = [float(v)]
                buf["edge"].append(ebuf)
            yield buf

    def convert(self, spark_df, graph_name):
        global NODE_DICT
        global TYPE_MAP
        global NODE_ATTR_DICT
        global EDGE_ATTR_DICT
        assert isinstance(self.schema_info, SchemaInfo)

        # data = spark_df.toPandas()
        data = spark_df

        data = self.preprocess(data)
        graph_path = os.path.join(self.save_dir, graph_name)
        os.makedirs(graph_path, exist_ok=True)
        meta_file = os.path.join(graph_path, graph_name + "_meta.json")
        data_file = os.path.join(graph_path, graph_name + "_data.json")
        binary_file = os.path.join(graph_path, graph_name + "_data.dat")

        # write graph meta
        print("writing graph meta...")
        meta = self._get_schema_meta()
        with open(meta_file, "w") as f:
            f.write(json.dumps(meta) + "\n")

        # write graph json
        print("writing graph json...")
        G = self.parse_df2nx(data)
        with open(data_file, "w") as f:
            for node_json in self.parse_df2euler(G):
                f.write(json.dumps(node_json) + "\n")

        cacher = SchemaCacher()
        graph_meta = {"node_dict": NODE_DICT, "type_map": TYPE_MAP,
                "node_attr_dict": NODE_ATTR_DICT, "edge_attr_dict": EDGE_ATTR_DICT}
        cacher.cache(df=data, schema_id=self.schema_id, graph_name=graph_name, graph_meta=graph_meta)

        print("convert json to binary data...")
        cmd = "/root/anaconda2/bin/python2 -m euler.tools -c %s -i %s -o %s" % \
              (meta_file, data_file, binary_file)
        os.system(cmd)


if __name__ == "__main__":
    import pandas as pd
    from bson.objectid import ObjectId

    schema_dict2 = {
        "_id": ObjectId("5ce50ed10da1b05d5d7174db"),
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
                        "method": "index",
                        "extract": False,
                        "digit": "0"
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
    # db = pymongo.MongoClient(mongo_adr)["gcn_lab"]
    # db["schema"].save(schema_dict2)


    # pdf = pd.read_csv("../../20w_test.csv")
    pdf = pd.read_csv("20w_test.csv")
    writer = GraphWriter()

    writer.set_schema(schema_dict2)
    writer.convert(pdf, "graph_01")

    print(TYPE_MAP)
    print(NODE_ATTR_DICT)
    print(EDGE_ATTR_DICT)

    # pdf = writer.preprocess(pdf)
    # G = writer.parse_df2nx(pdf)
















