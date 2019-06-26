import json
import pandas as pd
import networkx as nx
from tqdm import tqdm

NODE_DICT = {}  # global node index dict
TOP_NODE_ID = 0  # current node index


def load_data(data):
    global TOP_NODE_ID

    data = data.toPandas()

    G = nx.Graph()
    # ----- add node ----- #
    # add challenge
    challenge_list = data["challenge"].tolist()
    for n in challenge_list:
        if n not in NODE_DICT:
            NODE_DICT[n] = TOP_NODE_ID
            G.add_node(TOP_NODE_ID, t=0)
            TOP_NODE_ID += 1
    # add captcha
    captcha_list = data["captcha_id"].tolist()
    for n in captcha_list:
        if n not in NODE_DICT:
            NODE_DICT[n] = TOP_NODE_ID
            G.add_node(TOP_NODE_ID, t=1)
            TOP_NODE_ID += 1
    # add UA
    ua_list = data["UA"].tolist()
    for n in ua_list:
        if n not in NODE_DICT:
            NODE_DICT[n] = TOP_NODE_ID
            G.add_node(TOP_NODE_ID, t=2)
            TOP_NODE_ID += 1

    # ----- add edge ----- #
    for i, row in data.iterrows():
        # challenge <-> captcha
        src = NODE_DICT[row["challenge"]]
        tar = NODE_DICT[row["captcha_id"]]
        G.add_edge(src, tar, t=0)
        # challenge <-> ua
        src = NODE_DICT[row["challenge"]]
        tar = NODE_DICT[row["UA"]]
        G.add_edge(src, tar, t=1)
    return G


def gen_meta():
    out = open('test_meta.json', 'w')
    meta = {
        "node_type_num": 3,
        "edge_type_num": 2,
        "node_uint64_feature_num": 0,
        "node_float_feature_num": 1,
        "node_binary_feature_num": 1,
        "edge_uint64_feature_num": 0,
        "edge_float_feature_num": 0,
        "edge_binary_feature_num": 0
    }
    out.write(json.dumps(meta) + '\n')
    out.close()


# TODO: convert to euler block
def convert_data(G):
    out = open('test_data.json', 'w')

    for node in tqdm(G.nodes()):
        buf = dict()
        buf["node_id"] = node
        buf["node_type"] = G.node[node]["t"]
        buf["node_weight"] = 1
        buf["neighbor"] = {"0": {}, "1": {}}
        for n in G[node]:
            buf["neighbor"][str(G[node][n]["t"])][str(n)] = 1
        buf["uint64_feature"] = {}
        buf["float_feature"] = {}
        buf["binary_feature"] = {}

        buf["uint64_feature"] = {}
        buf["float_feature"][0] = [0.5]
        buf["binary_feature"][0] = str(G.node[node]["t"])

        buf["edge"] = []
        for tar in G[node]:
            ebuf = dict()
            ebuf["src_id"] = node
            ebuf["dst_id"] = tar
            ebuf["edge_type"] = G[node][tar]["t"]
            ebuf["weight"] = 1
            ebuf["uint64_feature"] = {}
            ebuf["float_feature"] = {}
            ebuf["binary_feature"] = {}
            buf["edge"].append(ebuf)
        out.write(json.dumps(buf) + '\n')
    out.close()


if __name__ == "__main__":
    pass

