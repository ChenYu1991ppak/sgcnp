import pandas as pd
from collections import OrderedDict

from ...db.redis_graph import RedisGraphSampler


class GraphReader(object):

    def __init__(self, redis_host, redis_port, redis_pwd):
        # GraphDB connection config
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_pwd = redis_pwd

        self.sampler = None
        self.map_client = None

        self._initialize()

    def _initialize(self):
        self.sampler = RedisGraphSampler(self.redis_host, self.redis_port, self.redis_pwd)
        # self.map_client = redis.StrictRedis(host=self.redis_host, port=self.redis_port, password=self.redis_pwd, db=6)

    def sample_graph(self, graph_name, graph_num, *degree):
        """ sample sub-graph from a whole graph.

            input:
                graph_name:
                num:
                degree:
            return:
                nodes_pds: a pandasDF dict saving all nodes in sub-graph divided by type.
                edges_pds: a pandasDF dict saving all edges in sub-graph divided by type.
        """
        # sample nodes and edges set from large graph
        node_dict, edge_dict = self.sampler.sample(graph_name, graph_num, *degree)
        node_pds = OrderedDict()
        edge_pds = OrderedDict()
        for node_label, node_items in node_dict.items():
            node_pds[node_label] = pd.DataFrame(node_items)
        for edge_label, edge_items in edge_dict.items():
            edge_pds[edge_label] = pd.DataFrame(edge_items)

        return node_pds, edge_pds


if __name__ == "__main__":
    reader = GraphReader("127.0.0.1", 6379, None)
    node_pds, edge_pds = reader.sample_graph("20w_test", 1, 10, 10)
    # print(node_pds["N_Challenge"]["request_time"])


