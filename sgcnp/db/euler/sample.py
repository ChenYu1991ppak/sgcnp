import os
import requests
import pymongo
import json
import pandas as pd
import time
import numpy as np
from collections import OrderedDict
import tf_euler
import tensorflow as tf

from config import euler_graph_cfg, database


class Euler(object):

    def __init__(self, graph_path):
        self.session = tf.Session()
        self.graph_path = graph_path

        self._initialize()

    def _initialize(self):
        print(self.graph_path)
        tf_euler.initialize_embedded_graph(self.graph_path)

    def sample_fanout(self, nodes, edge_types, counts, default_node=-1):
        results = tf_euler.sample_fanout(
            nodes,
            [edge_types for _ in nodes],
            counts,
            default_node
        )
        neighbors, weights, etypes = results
        return self.session.run([neighbors, weights, etypes])

    def sample_subgraph_fanout(self, batch_size, degrees, edge_types):
        tf.reset_default_graph()
        self.session = tf.Session()

        # sample source nodes at random
        src_nodes = tf_euler.sample_node(batch_size, tf_euler.ALL_NODE_TYPE)
        src_nodes = self.session.run(src_nodes)

        batchs = []

        for src_node in src_nodes:
            cur_nodes = np.asarray([src_node]).astype(np.int64)
            neighbors, weights, etypes = self.sample_fanout(cur_nodes, edge_types, degrees)
            sample = [[neighbors[0][0]]]
            for s in zip(neighbors[1:], weights, etypes):
                sample.append([i.tolist() for i in s])
            batchs.append(sample)
        return batchs

    def get_node_type(self, nodes):
        nodes = np.asarray(nodes).astype(np.int64)
        ntype = tf_euler.get_sparse_feature(
            nodes,
            [0],
        )
        return self.session.run(ntype[0]).values.tolist()

    def _get_dense_feature(self, nodes, feature_ids):
        f = tf_euler.get_dense_feature(
            nodes,
            feature_ids,
            [1],
        )
        return self.session.run(f[0]).tolist()

    def _get_binary_feature(self, nodes, feature_ids):
        f = tf_euler.get_binary_feature(
            nodes,
            feature_ids
        )
        return [item.decode("utf-8") for item in self.session.run(f[0]).tolist()]

    def get_node_feature(self, nodes, attr_meta):
        features = {}
        nodes = np.asarray(nodes).astype(np.int64)
        # print(attr_meta)
        for name, attr in attr_meta.items():
            attr_type = attr[0]
            attr_ord = attr[1]
            f = None
            if attr_type == "float":
                f = self._get_dense_feature(nodes, [attr_ord])
                f = [i[0] for i in f]
            elif attr_type == "binary":
                f = self._get_binary_feature(nodes, [attr_ord])
            features[name] = f
        return features

    def _get_edge_dense_feature(self, edges, feature_ids):
        f = tf_euler.get_edge_dense_feature(
            edges,
            feature_ids,
            [1],
        )
        return self.session.run(f[0]).tolist()

    def _get_edge_binary_feature(self, edges, feature_ids):
        f = tf_euler.get_edge_binary_feature(
            edges,
            feature_ids
        )
        return self.session.run(f[0]).tolist()


    def get_edge_feature(self, edges, eattr_meta):
        features = {}
        edges = np.asarray(edges).astype(np.int64)
        for name, eattr in eattr_meta.items():
            eattr_type = eattr[0]
            eattr_ord = eattr[1]
            f = None
            if eattr_type == "float":
                f = self._get_edge_dense_feature(edges, [eattr_ord])
                f = [i[0] for i in f]
            elif eattr_type == "binary":
                f = self._get_edge_binary_feature(edges, [eattr_ord])
            features[name] = f
        return features


class EulerGraphSampler(object):

    def __init__(self, mongo_url=database, cfg=euler_graph_cfg):
        self.graph_dir = os.path.join(os.getcwd() + "/../../", cfg["graph_dir"])

        self.cur_graph_path = None
        self.graph_meta = None

        self.mongo_url = mongo_url
        self.db = pymongo.MongoClient(mongo_url)["gcn_lab"]

        self.node_dict = None
        self.type_map = None
        self.node_attr_dict = None
        self.edge_attr_dict = None

        self.euler = None

        self.ntype_name_map = {}
        self.etype_name_map = {}

        self._initialize()

    def _initialize(self):
        self.db = pymongo.MongoClient(self.mongo_url)["gcn_lab"]

        # TODO: start python2 server
        pass

    def _init_graph(self, graph_path):
        if self.cur_graph_path != graph_path:
            print("initializing graph...")
            self.euler = Euler(graph_path)

            print("initialization finish.")
            print("graph path: %s" % graph_path)
            graph_name = os.path.basename(graph_path)
            self.graph_meta = self.db["graph_meta"].find_one({"_id": graph_name})
            self.cur_graph_path = graph_path

            graph_meta = self.db["graph_meta"].find_one({"_id": graph_name})
            self.node_dict = graph_meta["node_dict"]
            self.type_map = graph_meta["type_map"]
            self.node_attr_dict = graph_meta["node_attr_dict"]
            self.edge_attr_dict = graph_meta["edge_attr_dict"]

            for k, v in self.type_map.items():
                if k[:2] == "N_":
                    self.ntype_name_map[v] = k
                elif k[:2] == "E_":
                    self.etype_name_map[v] = k

    def _random_sample(self, graph_name, batch_size, *degrees):
        graph_path = os.path.join(self.graph_dir, graph_name)
        self._init_graph(graph_path)
        # get euler sample results
        edge_cnt = 0
        for k in self.type_map.keys():
            if k[:2] == "E_":
                edge_cnt += 1

        results = self.euler.sample_subgraph_fanout(batch_size, degrees, [i for i in range(edge_cnt * 2)])
        # push results into DF
        records = []
        nodes_idx = []
        for result in results:
            src_set = result[0]
            nodes_idx.extend(src_set)
            for i, tar_set in enumerate(result[1:]):
                d = degrees[i]
                for j, src in enumerate(src_set):
                    for idx in range(j * d, (j + 1) * d):
                        tar = tar_set[0][idx]
                        weight = tar_set[1][idx]
                        edge_type = tar_set[2][idx]
                        r = {"src": src, "tar": tar, "weight": weight, "etype": edge_type}
                        records.append(r)
                src_set = tar_set[0]
                nodes_idx.extend(src_set)
        epdf = pd.DataFrame(records)
        epdf = epdf.drop_duplicates(subset=["src", "tar"], keep='first').reset_index(drop=True)
        npdf = pd.DataFrame([{"NodeID": n} for n in nodes_idx])
        npdf = npdf.drop_duplicates(subset=["NodeID"], keep='first').reset_index(drop=True)

        nodes = npdf["NodeID"].tolist()
        ntype = self.euler.get_node_type(nodes)

        npdf["ntype"] = pd.Series(ntype)

        return npdf, epdf

    def _get_feature(self, npdf, epdf):
        ntype = npdf["ntype"].drop_duplicates().tolist()
        node_pds = OrderedDict()
        for t in ntype:
            pdf = npdf[npdf["ntype"] == t].copy().reset_index(drop=True)
            attr_meta = self.node_attr_dict[str(t)]
            nodes = pdf["NodeID"].tolist()
            nfeature = self.euler.get_node_feature(nodes, attr_meta)
            for k, seri in nfeature.items():
                pdf[k] = pd.Series(seri)
            pdf["ntype"] = self.ntype_name_map[t]
            node_pds[self.ntype_name_map[t]] = pdf

        etype = epdf["etype"].drop_duplicates().tolist()
        edge_pds = OrderedDict()
        for t in etype:
            pdf = epdf[epdf["etype"] == t].copy().reset_index(drop=True)
            eattr_meta = self.edge_attr_dict[str(t)]
            edges = list(zip(pdf["src"].tolist(), pdf["tar"].tolist(), [t] * len(pdf['src'].tolist())))
            efeature = self.euler.get_edge_feature(edges, eattr_meta)
            for k, seri in efeature.items():
                pdf[k] = pd.Series(seri)
            pdf["etype"] = self.etype_name_map[t]
            pdf["SrcNodeID"] = pdf["src"]
            pdf["DestNodeID"] = pdf["tar"]
            edge_pds[self.etype_name_map[t]] = pdf

        return node_pds, edge_pds

    def sample(self, graph_name, batch_size, *degrees):
        npdf, epdf = self._random_sample(graph_name, batch_size, *degrees)

        node_pds, edge_pds = self._get_feature(npdf, epdf)
        return node_pds, edge_pds


if __name__ == "__main__":
    import numpy as np
    s = EulerGraphSampler()
    e, n = s.sample("graph_01", 1, 10, 10, 10)
    # print(e)
    # print(n)





