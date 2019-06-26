from collections import OrderedDict


class NodeInfo(object):

    def __init__(self, node_config):
        # type id is unique for schema
        self._type_id = node_config["id"]
        # node type is unique for graph
        self._node_type = node_config["type"]
        self._node_attr = node_config["attributes"]
        if "copied" in node_config.keys():
            self._copied = node_config["copied"]
        else:
            self._copied = False

    @property
    def type_id(self):
        return self._type_id

    @property
    def type(self):
        return "N_" + self._node_type

    @property
    def is_copy(self):
        return self._copied

    @property
    def attrs(self):
        for attr in self._node_attr:
            is_index = attr["is_index"] if "is_index" in attr.keys() else False
            meta = None
            digit = None
            if attr["method"] == "index":
                meta = attr["embedding"]
                digit = attr["digit"]
            if attr["method"] == "normalize":
                meta = attr["normalize_stat"]

            yield {"name": attr["name"],
                   "source": attr["source"],
                   "method": attr["method"],
                   "is_index": is_index,
                   "meta": meta,
                   "digit": digit
                   # "multi_items": multi_items
                   }


class EdgeInfo(object):

    def __init__(self, edge_config):
        self._edge_type = edge_config["type"]
        self._edge_src_type = edge_config["source"]
        self._edge_tar_type = edge_config["target"]
        self._edge_attr = edge_config["attributes"]

        if "multi" in edge_config.keys():
            self._multi = edge_config["multi"]
        else:
            self._multi = None

    @property
    def type(self):
        return "E_" + self._edge_type

    @property
    def src_type_id(self):
        return self._edge_src_type

    @property
    def tar_type_id(self):
        return self._edge_tar_type

    @property
    def multi(self):
        return self._multi

    @property
    def attrs(self):
        for attr in self._edge_attr:
            meta = None
            digit = None
            if attr["method"] == "index":
                meta = attr["embedding"]
                digit = attr["digit"]
            if attr["method"] == "normalize":
                meta = attr["normalize_stat"]

            yield {"name": attr["name"],
                   "source": attr["source"],
                   "method": attr["method"],
                   "meta": meta,
                   "digit": digit
                   }


class SchemaInfo(object):

    def __init__(self, schema_dict):
        schema_dict = OrderedDict(schema_dict)

        self._schema_id = schema_dict["_id"]

        self._node_dict = schema_dict["nodes"]
        self._edge_dict = schema_dict["edges"]

        self._node_info = []
        self._edge_info = []

        self._initialize()

    def _initialize(self):
        for cfg in self._node_dict:
            self._node_info.append(NodeInfo(cfg))
        for cfg in self._edge_dict:
            self._edge_info.append(EdgeInfo(cfg))

    @property
    def schema_id(self):
        return self._schema_id

    @property
    def node_info_list(self):
        return self._node_info

    @property
    def edge_info_list(self):
        return self._edge_info


if __name__ == "__main__":
    pass
