import numpy as np
import json


class SchemaCacher(object):
    '''
    图的schema, 里面不仅有结构和feature, 而且还缓存了embedding方法需要的映射表, normalize 需要的统计量
    另外, 我们对平台的schema能进行的操作进行限制
    1. schema只允许新建或者复制旧的schema作为参考, 不允许修改
    2. schema允许删除, 删除后只是限制了不允许再根据这个schema新建图数据, 但旧的图数据和model还是能使用的

    schema 数据组织还是选择将index表单独独立的方法, redis不支持map<key, map>的方式, hmset效率更高
    cache 会产生如下几条key
    1. K: schema_<name>_config
       V: json_string

    2. K: schema_<name>_key_<key>_index
       V: {'_default_value_magic_key':0,
            'key1':1, ...}
    '''

    def __init__(self, client):
        self.client = client

    def cache(self, df, schemacfg, graphdataset=None, if_empty_insert=False):
        '''

        :param df:
        :param schema:
        :param if_empty_insert: 只有为空的时候才cache
        :return: 是否更新成功
        '''
        schemaname = schemacfg["_id"]

        schema_redis_key = "schema_%s" % schemaname
        if if_empty_insert and self.client.get(schema_redis_key) is not None:
            return False

        if graphdataset is not None:
            self.bind_graphdataset(schemaname, graphdataset)

        attrs = sum([e['attributes'] for e in schemacfg['edges'] + schemacfg['nodes']], [])
        # 做normalize 统计表
        normalize_source = list(
            set(map(lambda x: x['source'], filter(lambda x: x.get('method', '') == 'normalize', attrs))))
        for normalize_key in normalize_source:
            if not np.issubdtype(df[normalize_key].dtype, np.number):
                df[normalize_key] = df[normalize_key].apply(self._obj2number)
        normalize_map = df[normalize_source].describe().to_dict()
        # 做embedding 映射表
        embedding_map = {}
        indexed_source = set(map(lambda x: x['source'], filter(lambda x: x.get('method', '') == 'index', attrs)))
        for indexed_key in indexed_source:
            redis_indexed_key = self._fetch_schema_idx_redis_key(schemaname, indexed_key)
            embedding_map[indexed_key] = redis_indexed_key
            _ = self._fetch_feature_index(df[indexed_key], self.client, redis_indexed_key, forceinsert=True)
        # 把所有映射表写入schema redis
        for attr in attrs:
            if attr['method'] == 'normalize':
                attr['normalize_stat'] = normalize_map[attr['source']]
            elif attr['method'] == 'index':
                attr['embedding'] = embedding_map[attr['source']]
            else:
                # TODO: raise
                pass
        self.client.set(self._fetch_schema_redis_key(schemaname), json.dumps(schemacfg))

    def bind_graphdataset(self, schemaname, graphdataset):
        '''
        绑定图数据
        :param name:
        :return:
        '''
        return self.client.hset("schema_graphdataset_mapper", graphdataset, schemaname)

    def fetch_schema(self, graphdataset):
        '''
        获取图数据对应的schema
        :param graphdataset:
        :return:
        '''
        return self.client.hget("schema_graphdataset_mapper", graphdataset).decode('utf-8')

    @staticmethod
    def _obj2number(x):
        try:
            return float(x) if x != 'None' else 0
        except:
            return 0

    def get(self, graphdataset):

        schemaname = self.fetch_schema(graphdataset)
        # decode
        schema_redis_key = self._fetch_schema_redis_key(schemaname)
        schemacfg = json.loads(self.client.get(schema_redis_key).decode("utf-8"))
        attrs = sum([e['attributes'] for e in schemacfg['edges'] + schemacfg['nodes']], [])
        for attr in attrs:
            if attr['method'] == 'index':
                attr['embedding'] = self.redis2decode(self.client.hgetall(attr['embedding']))
        return schemacfg

    def _delete(self, schemaname):
        '''
        考虑schema可以被用户删除, 但是会被model和graphdataset依赖,
        默认不要调用该方法, 除非有机制判断现行不存在依赖该schema的数据和模型
        :return:
        '''
        schema_redis_key = "schema_%s*" % schemaname
        for k in self.client.keys(schema_redis_key):
            self.client.delete(k)
        return True

    @staticmethod
    def redis2decode(_dict):
        return dict([(k.decode('utf-8'), int(v)) for k, v in _dict.items()])

    @staticmethod
    def _fetch_schema_redis_key(schemaname):
        return "schema_%s_config" % schemaname

    @staticmethod
    def _fetch_schema_idx_redis_key(schemaname, index_key):
        return "schema_%s_key_%s_idx" % (schemaname, index_key)

    @staticmethod
    def _fetch_feature_index(features, client, redis_key, forceinsert=False):
        '''
        embedding 是个复杂的部分, 有非常多的方法, 这里我们用一种非常理想的场景来组织这个模块
        1) one hot, index, 还有模型内的embedding layer, 我们统一用一种 index + LabelEmbedding(cfg=bits) 来实现
        2) 数据本身是由数据源 + schema组织出来的, 所以我们要保证公用不同schema但数据源不同的图数据集能适用同一个模型(模型也是和schema帮定的), so.. 我们记录在 schema_featurename 表中
        3) 我们对需要做index的字段是有要求的, eg: 较为稳定, 不太会扩增, 这样一可以在第一次建数据的时候就可以覆盖绝大部分 样本, 二: 不稳定的index字段, 更应该作为节点类型去考虑, 这样更能表达出节点的结构信息而不是出现频次非常低的feature
        4) 综上, 给出forceinsert字段, 如果为True, 那么每次查询index的时候,就会往schema进行同步, 否则只在该表不存在的时候进行同步;
        5) 对于schema中不存在的index, 默认返回 0, 相当于每张index表默认的第一个映射关系就留给未知数据

        :param features:
        :param client:
        :param schema:
        :param featurename:
        :param forceinsert:
        :return:
        '''
        db = redis_key
        DEFAULT_VALUE_MAGIC_KEY = "_default_value_magic_key"
        idx_dict_redis = dict([(k.decode('utf-8'), int(v)) for k, v in client.hgetall(db).items()])
        is_first_insert = idx_dict_redis.get(DEFAULT_VALUE_MAGIC_KEY, None) is None
        is_update_idx_in_redis = (forceinsert or is_first_insert)
        lastidx, idx_dict_ext, feature_idxs = len(idx_dict_redis), {}, []
        if is_first_insert:
            idx_dict_ext[DEFAULT_VALUE_MAGIC_KEY] = 0
            lastidx = 1
        for f in features:
            f = str(f)
            feature_idx = idx_dict_redis.get(f, idx_dict_ext.get(f, None))
            if feature_idx is None:
                if is_update_idx_in_redis:
                    feature_idx = lastidx
                    idx_dict_ext[f] = feature_idx
                    lastidx += 1
                else:
                    feature_idx = idx_dict_redis.get(DEFAULT_VALUE_MAGIC_KEY,
                                                     idx_dict_ext.get(DEFAULT_VALUE_MAGIC_KEY, None))
            feature_idxs.append(feature_idx)
        del idx_dict_redis
        if is_update_idx_in_redis and len(idx_dict_ext) != 0:
            client.hmset(db, idx_dict_ext)

        return feature_idxs


# gee007 spark环境测试通过
if __name__ == "__main__":
    pass

