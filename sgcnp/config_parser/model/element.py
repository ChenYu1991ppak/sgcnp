from .types import ElementType


class ElementInfo(object):

    def __init__(self, cfg):
        # ---------- in trainer --------- #
        # elem index in a specific trainer
        self.__dict__.update({
            "index": cfg["idx"],
            "type": None,
            "order": None,
            "front": cfg["front"],
            # output indexes in trainer
            "back": [],
            # input sources and merging methods of the layer
            # ---------- in element --------- #
            "elem_type": cfg["elem_type"],
            # info used to construct the element
            "inner": cfg["inner"],
            # output indexes
            "output_idx": cfg["output_idx"],
            # parameter used to init
            "parameter": cfg["parameter"],
        })

        self._initialize()

    def _initialize(self):
        self._parse_idx()
        self._expose_outs_idx()

    def _parse_idx(self):
        et, order = self.index.split("_")
        self._type, self._order = et, int(order)

    def _expose_outs_idx(self):
        for out in self.output_idx:
            self.back.append(self.index + "." + out)


if __name__ == "__main__":
    pass
