import redis
import struct
from tqdm import tqdm
import math
import pandas as pd
from pandas.api import types as pdtypes


class Type:
    NULL = 0
    BOOL = 1
    NUMERIC = 2
    STRING = 3


def pack_null_func(attr):
    return struct.pack("=B", Type.NULL)


def pack_numeric_func(attr):
    return struct.pack("=Bd", Type.NUMERIC, attr)


def pack_string_func(attr):
    if isinstance(attr, float) and math.isnan(attr):
        attr = ""
    if attr is None:
        attr = ""
    encoded_str = attr.encode('utf-8')
    return struct.pack("=B%ds" % (len(encoded_str) + 1), Type.STRING, encoded_str)


def pack_bool_func(attr):
    return struct.pack("=B?", Type.BOOL, attr)


class Packer(object):
    @staticmethod
    def derive_pack_func(col):
        if pdtypes.is_string_dtype(col):
            return pack_string_func
        elif pdtypes.is_bool_dtype(col):
            return pack_bool_func
        elif pdtypes.is_numeric_dtype(col):
            return pack_numeric_func
        else:
            return pack_null_func

    @staticmethod
    def pack_header(props, _type):
        props = [p.encode('utf-8') for p in props]
        prop_count = len(props)
        _type = _type.encode('utf-8')
        fmt = "".join(['=%dsI' % (len(_type) + 1)] + [("%ds" % (len(prop) + 1)) for prop in props])
        return struct.pack(fmt, *[_type, prop_count] + props)

    @staticmethod
    def pack_edge(src, tar):
        return struct.pack("=QQ", src, tar)


class Component(object):

    def __init__(self, part, ptype):
        """ The base component classes that compose a graph
            including Node and Edge.

            input:
                part : Pandas DF, or CSV file path.
                ptype: component name.
        """
        self.part = part
        self.ptype = ptype

        self.df = None  # Pandas DF form of component data
        self.prop_cols = []  # properties of component data
        self.bit_prop_cols = []  # bit form of properties used to transmitting
        self.cnt = 0  # number of data

        self.header = None
        self.token = None

        self._initialize()

    def _initialize(self):
        self.df = pd.read_csv(self.part) if type(self.part) is str else self.part
        self.cnt = self.df.shape[0]

        self._get_properties_keys()
        self._pack_properties()

    def _get_properties_keys(self):
        for col_key in self.df.keys().values:
            self.prop_cols.append(col_key)

    def _pack_properties(self):
        for col_key in self.prop_cols:

            bit_col = "%s.bit" % col_key
            packfunc = Packer.derive_pack_func(self.df[col_key])

            self.df[bit_col] = self.df[col_key][:].apply(packfunc)
            self.bit_prop_cols.append(bit_col)

    def _derive_header(self):
        self.header = Packer.pack_header(self.prop_cols, self.ptype)

    def _derive_token(self):
        pass

    def yield_data_bits(self, split=1000000):
        pass


class NodeComponent(Component):

    def __init__(self, part, ptype):
        super(NodeComponent, self).__init__(part, ptype)
        self._derive_header()
        self._derive_token()

    def _derive_token(self):
        self.tokens = [self.header + b''.join(e) for e in self.df[self.bit_prop_cols].values]

    def yield_data_bits(self, split=100000):
        for idx in range(0, self.cnt, split):
            labels = self.tokens[idx: idx+split]
            # The first list have 4 elements represent to number of node, relation, label, reltype
            # Second is list of labels packed, and third is reltype packed
            args = [self.cnt, 0, len(labels), 0] + labels + []
            yield args


class EdgeComponent(Component):

    def __init__(self, part, ptype):
        super(EdgeComponent, self).__init__(part, ptype)
        self.src = self.prop_cols[0]
        self.tar = self.prop_cols[1]
        self._derive_header()
        self._derive_token()

    def _derive_token(self):
        # generate unique identification of relation
        self.df['_src_tar_bit'] = [Packer.pack_edge(s, t) for s, t in self.df[[self.src, self.tar]].values]
        self.tokens = [self.header +
                       b''.join(e) for e in self.df[['_src_tar_bit'] + self.bit_prop_cols].values]

    def yield_data_bits(self, split=100000):
        for idx in range(0, self.cnt, split):
            reltypes = self.tokens[idx:idx+split]
            # The first list have 4 elements represent to number of node, relation, label, reltype
            # Second is list of labels packed, and third is reltype packed
            args = [0, self.cnt, 0, len(reltypes)] + [] + reltypes
            yield args


class RedisBufferSender(object):
    def __init__(self, client,
                 graph_name, split=100000):
        self.client = client
        self.first_insert = True
        self.graph_name = graph_name
        self.split = split

        # module_list = self.client.execute_command("MODULE LIST")
        # if not any(b'graph' in module_description for module_description in module_list):
        #     print("RedisGraph module not loaded on connected server.")
        #     exit(1)

    def insert(self, components):
        """ bulk insert """
        nodes_created = 0
        relations_created = 0

        if not isinstance(components, list):
            components = [components]
        for component in components:
            for args in component.yield_data_bits(split=self.split):
                if self.first_insert:
                    args.insert(0, "BEGIN")
                    self.first_insert = False
                result = self.client.execute_command("GRAPH.BULK", self.graph_name,  *args)
                stats = result.split(', '.encode())
                nodes_created += int(stats[0].split(' '.encode())[0])
                relations_created += int(stats[1].split(' '.encode())[0])
        return nodes_created, relations_created




