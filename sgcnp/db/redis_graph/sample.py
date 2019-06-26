import redis
import numpy as np
import json
from collections import OrderedDict


class RedisGraphSampler(object):

    def __init__(self, redis_host, redis_port, redis_pwd):
        self.client = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_pwd)

    def _random_sample(self, graph_name, graph_num, *degrees):
        # sample source nodes randomly
        node_num = self.client.execute_command("GRAPH.QUERY",
                                               graph_name,
                                               "MATCH (r) RETURN count(r)")[0][1][0]
        src_nodes = np.random.choice(np.arange(node_num), size=graph_num)
        src_nodes_str = ""
        for node_id in src_nodes:
            if src_nodes_str != "":
                src_nodes_str += ","
            src_nodes_str += str(node_id)
        degrees_str = ""
        for d in degrees:
            if degrees_str != "":
                degrees_str += ","
            degrees_str += str(d)

        objects = self.client.execute_command("GRAPH.SAMPLE",
                                              graph_name,
                                              src_nodes_str,
                                              degrees_str)
        objects = [json.loads(r.decode()) for r in objects]
        return objects

    @staticmethod
    def _split_result(objects):
        node_dict = OrderedDict()
        edge_dict = OrderedDict()
        for obj in objects:
            if "ntype" in obj.keys():
                node_type = obj["ntype"]
                if node_type not in node_dict.keys():
                    node_dict[node_type] = [obj]
                else:
                    node_dict[node_type].append(obj)
            if "etype" in obj.keys():
                edge_type = obj["etype"]
                if edge_type not in edge_dict.keys():
                    edge_dict[edge_type] = [obj]
                else:
                    edge_dict[edge_type].append(obj)
        return node_dict, edge_dict

    def sample(self, graph_name, graph_num, *degrees):
        objects = self._random_sample(graph_name, graph_num, *degrees)
        node_dict, edge_dict = self._split_result(objects)
        return node_dict, edge_dict


if __name__ == "__main__":
    sampler = RedisGraphSampler("127.0.0.1", 6379, None)
    subgraph = sampler.sample("test", 1, 10, 10, 10)
    print(subgraph[1])
