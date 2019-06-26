
class BaseInfo(object):

    def __init__(self, cfg):
        self.__dict__.update({
            "index": cfg["idx"],
            "type": None,
            "order": None,

            "front": cfg["front"] if "front" in cfg.keys() else [None],
            "back": cfg["back"] if "back" in cfg.keys() else [cfg["idx"]],

            "obj_class": cfg["class"],
            "parameter": cfg["parameter"]

        })
        self._initialize()

    def _initialize(self):
        pass

class GCNLayerInfo(BaseInfo):

    def __init__(self, cfg):
        super(GCNLayerInfo, self).__init__(cfg=cfg)

    def _initialize(self):
        self._parse_idx()

    def _parse_idx(self):
        t, order = self.index.split("_")
        self._type, self._order = t, int(order)


class LossInfo(BaseInfo):

    def __init__(self, cfg):
        super(LossInfo, self).__init__(cfg=cfg)

        self._initialize()

    def _initialize(self):
        self._parse_idx()

    def _parse_idx(self):
        t, order = self.index.split("_")
        self._type, self._order = t, int(order)


class SourceInfo(BaseInfo):

    def __init__(self, cfg):
        super(SourceInfo, self).__init__(cfg=cfg)

    def _initialize(self):
        self._parse_idx()

    def _parse_idx(self):
        t, order = self.index.split("_")
        self._type, self._order = t, int(order)
