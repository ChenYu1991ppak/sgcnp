# -*- coding:utf-8 -*-
from ..model.loss.func import loss_func_map
class ConfigTransformer(object):
    '''
    对前端传过来的config做适配
    '''

    def __init__(self, web_cfg):
        self.web_cfg = web_cfg

        self.dataset_info = {}
        self.elements_config = {}
        self.computation_config = {}
        self.elements_output = {}
        self.block_set = []
        self.block_map = {}

        self.elem_set = []
        self.output_map = {}
        self.loss_map = loss_func_map

        self._initialize()

    def _initialize(self):
        for ds in self.web_cfg["dataset"]:
            self.dataset_info[ds["name"]] = ds

    @staticmethod
    def input_key_convert(input_key):
        cls, idx = input_key.split("_")
        if cls == "Input":
            return "input_" + str(int(idx) - 1)
        else:
            return input_key

    def layer_cfg_convert(self, layer_cfg):
        layers_dict = dict()
        layers_dict["class"] = layer_cfg["type"]
        layers_dict["parameter"] = layer_cfg["config"] if "config" in layer_cfg.keys() else {}
        layers_dict["front"] = []
        for item in layer_cfg["input"]:
            layers_dict["front"].append([layer_cfg["merge"], self.input_key_convert(item)])
        return layers_dict

    def parse_blocks2elements(self):
        blocks_cfg = self.web_cfg["blocks"]

        for block_cfg in blocks_cfg:
            block_layers_cfg = block_cfg["graph"]

            block_layers_dict = {}
            block_outputs = []
            for layer_cfg in block_layers_cfg:

                layer_id = layer_cfg["id"]
                layer_type = layer_cfg["type"]
                if layer_type != "Input" and layer_type != "Output":
                    # init normal layer
                    block_layers_dict[layer_id] = self.layer_cfg_convert(layer_cfg)
                elif layer_type == "Output":
                    # record output of block
                    block_outputs.append(self.input_key_convert(layer_cfg["input"][0]))

            block_num = len(self.block_set)
            net_name = "net_" + str(block_num + 1)
            self.elements_config[net_name] = block_layers_dict
            self.elements_output[net_name] = block_outputs
            self.block_set.append(net_name)
            self.block_map[block_cfg["name"]] = net_name

        return self.elements_config, self.elements_output, self.block_set

    def filter_src_input(self, input_list):
        input_list_new = []
        for item in input_list:
            if isinstance(item, list):
                if item[1].split("_")[0] == "src" and item[1].split(".")[-1] == "x":
                    input_list_new.append(item[1].split(".")[0] + ".edge_index")
                    input_list_new.append([None, item[1].split(".")[0] + ".x"])
                else:
                    input_list_new.append(item)
            else:
                input_list_new.append(item)
        return input_list_new

    def block_entity2elem(self, entity_cfg):
        elems_cfg_dict = {}

        entity_type = entity_cfg["type"]
        elems_cfg_dict["inner"] = self.block_map[entity_type]
        elems_cfg_dict["parameter"] = {"updated": "ALL", "written": "ALL"}
        elems_cfg_dict["elem_type"] = "model"

        input_list = []
        for k, v in entity_cfg["input"].items():
            input_list.append([int(k.split("_")[1]), v])
        input_list = sorted(input_list, key=lambda x: x[0])
        elems_cfg_dict["front"] = self.filter_src_input([[None, self.get_front_item(x[1])] for x in input_list])
        elems_cfg_dict["output_idx"] = self.elements_output[self.block_map[entity_type]]
        return elems_cfg_dict

    def dataset_entity2elem(self, entity_cfg):
        # add dataset into elements_config
        dataset_id = entity_cfg["id"]
        dataset_info = self.dataset_info[dataset_id]

        dataset_elem = {}
        dataset_elem["class"] = "base"
        dataset_elem["back"] = ["edge_index", "x", "y", "mask"]
        dataset_elem["parameter"] = {}
        dataset_elem["parameter"]["train_graph"] = dataset_info["train"]
        dataset_elem["parameter"]["test_graph"] = dataset_info["test"]
        dataset_elem["parameter"]["degree"] = dataset_info["config"]["degree"]
        dataset_elem["parameter"]["batch_size"] = dataset_info["config"]["batch_size"]

        ds_name = "dataset_" + str(len(self.block_set) + 1)
        dataset_elem["idx"] = entity_cfg["id"]
        self.elements_config[ds_name] = dataset_elem
        self.elements_output[ds_name] = ["edge_index", "x", "y", "mask"]
        self.block_set.append(ds_name)
        self.block_map[dataset_id] = ds_name
        # gen computation graph
        elems_cfg_dict = {}

        elems_cfg_dict["elem_type"] = "source"
        elems_cfg_dict["front"] = []
        elems_cfg_dict["inner"] = ds_name
        elems_cfg_dict["output_idx"] = ["edge_index", "x", "y", "mask"]
        elems_cfg_dict["parameter"] = {"updated": "ALL", "written": "ALL"}
        return elems_cfg_dict

    def loss_entity2elem(self, entity_cfg):
        loss_cfg = {}
        loss_cfg["class"] = entity_cfg["type"]
        loss_cfg["parameter"] = {}
        loss_cfg["idx"] = entity_cfg["id"]

        ls_name = "loss_" + str(len(self.block_set) + 1)
        self.elements_config[ls_name] = loss_cfg
        self.elements_output[ls_name] = []
        self.block_set.append(ls_name)
        self.block_map[entity_cfg["id"]] = ls_name
        # gen computation graph
        elems_cfg_dict = {}

        elems_cfg_dict["elem_type"] = "receiver"
        input_list = []
        input_list.append([None, self.get_front_item(entity_cfg["input"]["target"])])
        ds = self.get_front_item(entity_cfg["input"]["label"].split(".")[0]).split(".")[0]
        input_list.append([None, ds + ".y"])
        input_list.append(ds + ".mask")
        elems_cfg_dict["front"] = input_list
        elems_cfg_dict["inner"] = ls_name
        elems_cfg_dict["output_idx"] = []
        elems_cfg_dict["parameter"] = {}
        return elems_cfg_dict

    def get_front_item(self, input_item):
        if "." in input_item:
            front_entity, front_idx = input_item.split(".")
            if front_idx.split("_")[0] == "Output":
                # TODO
                front_idx = self.elements_output[self.block_map[front_entity.split("_")[0]]][int(front_idx.split("_")[1]) - 1]
                return self.output_map[front_entity] + "." + front_idx
            else:
                return self.output_map[front_entity] + "." + front_idx
        else:
            return self.output_map[input_item] + "." + input_item

    def layer_entity2elem(self, entity_cfg):
        layer_cfg = {}
        layer_cfg["class"] = entity_cfg["type"]
        layer_cfg["parameter"] = entity_cfg["config"] if "config" in entity_cfg.keys() else {}
        layer_input_list = []
        elem_input_list = []

        merge = entity_cfg["merge"] if "merge" in entity_cfg.keys() else None
        if merge is None:
            for i, input_item in enumerate(entity_cfg["input"]):
                layer_input_list.append([None, "input_" + str(i)])
                elem_input_list.append([None, self.get_front_item(input_item)])
        else:
            layer_input_list.append([merge, "input_0", "input_1"])
            input_item1 = self.get_front_item(entity_cfg["input"][0])
            input_item2 = self.get_front_item(entity_cfg["input"][1])
            elem_input_list.append([merge, input_item1, input_item2])

        layer_cfg["front"] = layer_input_list
        net_name = "net_" + str(len(self.block_set) + 1)
        self.elements_config[net_name] = {entity_cfg["id"]: layer_cfg}
        self.elements_output[net_name] = [entity_cfg["id"]]
        self.block_set.append(net_name)
        self.block_map[entity_cfg["id"]] = net_name

        #
        elems_cfg_dict = {}
        elems_cfg_dict["elem_type"] = "model"
        elems_cfg_dict["front"] = self.filter_src_input(elem_input_list)
        elems_cfg_dict["inner"] = net_name
        elems_cfg_dict["output_idx"] = [entity_cfg["id"]]
        elems_cfg_dict["parameter"] = {"updated": "ALL", "written": "ALL"}
        return elems_cfg_dict

    def parse_graph2computation(self):

        graph_cfg = self.web_cfg["graph"]
        computation_config = {}
        for entity_cfg in graph_cfg:
            entity_type = entity_cfg["type"]

            if entity_type in self.block_map.keys():
                elem_name = "mod_" + str(len(self.elem_set) + 1)
                self.output_map[entity_cfg["id"]] = elem_name
                computation_config[elem_name] = self.block_entity2elem(entity_cfg)
                self.elem_set.append(elem_name)
            elif entity_type == "Dataset":
                elem_name = "src_" + str(len(self.elem_set) + 1)
                self.output_map[entity_cfg["id"]] = elem_name
                computation_config[elem_name] = self.dataset_entity2elem(entity_cfg)
                self.elem_set.append(elem_name)
            elif entity_type in self.loss_map:
                elem_name = "rec_" + str(len(self.elem_set) + 1)
                self.output_map[entity_cfg["id"]] = elem_name
                computation_config[elem_name] = self.loss_entity2elem(entity_cfg)
                self.elem_set.append(elem_name)
            else:
                elem_name = "mod_" + str(len(self.elem_set) + 1)
                self.output_map[entity_cfg["id"]] = elem_name
                computation_config[elem_name] = self.layer_entity2elem(entity_cfg)
                self.elem_set.append(elem_name)

        self.computation_config["train_graph"] = computation_config

        task_dict = {}
        task_cfg = self.web_cfg["config"]
        task_dict["train_type"] = "graph"
        task_dict["optimizer_cfg"] = {}
        task_dict["optimizer_cfg"]["name"] = task_cfg["optimizer"]
        task_dict["optimizer_cfg"]["parameter"] = {}
        task_dict["optimizer_cfg"]["parameter"]["lr"] = task_cfg["lr"]
        task_dict["optimizer_cfg"]["parameter"]["momentum"] = task_cfg["momentum"]
        task_dict["optimizer_cfg"]["parameter"]["weight_decay"] = task_cfg["weight_decay"]
        task_dict["epoch"] = task_cfg["epoch"]
        task_dict["lr_decay_coef"] = task_cfg["lr_decay_coef"]
        task_dict["lr_decay_gap"] = task_cfg["lr_decay_gap"]
        task_dict["save_dir"] = ""
        task_dict["selected_loss"] = [self.output_map[item] for item in task_cfg["selected_loss"]]
        self.computation_config["train_task"] = task_dict

        return self.computation_config


if __name__ == "__main__":
    final_config = {
  "blocks": [],
  "graph": [
    {
      "id": "Dataset_1",
      "input": {},
      "type": "Dataset",
      "isBlock": True
    },
    {
      "id": "GCNConv_1",
      "input": [
        "Dataset_1.x"
      ],
      "type": "GCNConv",
      "config": {
        "out_channels": 32,
      },
      "isBlock": False,
      "merge": None
    },
    {
      "id": "CrossEntropy_1",
      "input": {
        "target": "GCNConv_1",
        "label": "Dataset_1.y"
      },
      "type": "CrossEntropy",
      "isBlock": True
    }
  ],
    "dataset": [
        {
            "name": "Dataset_1",
            "train": "Graph_Train_1", # if in test mode, this is None
            "test": "Graph_Test_1",  # optional, if not selected: None
            "config": {
                "degree": [10, 10, 10, 10, 10],
                "batch_size": 4,
            }
        }
    ],
    "config": {
        "optimizer": "SGD",  # 'Adam'
        "lr": 1e-2,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "epoch": 2000,
        "lr_decay_coef": 0.75,  # 0-1
        "lr_decay_gap": 200,  # int, < epoch
        "selected_loss": ["CrossEntropy_1"],  # select losses
    }
}

    final_config2 = {'folder': '/www/gcn-lab/instance/ffffff_5cee4e150da1b06840c4bb2e', 'blocks': [], 'graph': [{'id': 'Dataset_1', 'input': {}, 'type': 'Dataset', 'isBlock': True}, {'id': 'GCNConv_1', 'input': ['Dataset_1.x'], 'type': 'GCNConv', 'config': {'out_channels': 20, 'improved': False, 'bias': True}, 'isBlock': False, 'merge': None}, {'id': 'CrossEntropy_1', 'input': {'label': 'Dataset_1.y', 'target': 'GCNConv_1'}, 'type': 'CrossEntropy', 'isBlock': True}], 'dataset': [{'name': 'Dataset_1', 'train': 'ssssssss', 'test': 'ssssssss', 'config': {'degree': [10, 10, 10, 10, 10], 'batch_size': 4}}], 'config': {'optimizer': 'Adam', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0001, 'epoch': 2000, 'lr_decay_coef': 0.75, 'lr_decay_gap': 200, 'selected_loss': ['CrossEntropy_1']}}
    final_config3 = {'folder': '/www/gcn-lab/instance/ffffff_5cee4e150da1b06840c4bb2e', 'blocks': [], 'graph': [{'id': 'Dataset_1', 'input': {}, 'type': 'Dataset', 'isBlock': True}, {'id': 'GCNConv_1', 'input': ['Dataset_1.x'], 'type': 'GCNConv', 'config': {'out_channels': 20, 'improved': False, 'bias': True}, 'isBlock': False, 'merge': None}, {'id': 'CrossEntropy_1', 'input': {'label': 'Dataset_1.y', 'target': 'GCNConv_1'}, 'type': 'CrossEntropy', 'isBlock': True}], 'dataset': [{'name': 'Dataset_1', 'train': 'ssssssss', 'test': 'ssssssss', 'config': {'degree': [10, 10, 10, 10, 10], 'batch_size': 4}}], 'config': {'optimizer': 'Adam', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0001, 'epoch': 2000, 'lr_decay_coef': 0.75, 'lr_decay_gap': 200, 'selected_loss': ['CrossEntropy_1']}}

    tr = ConfigTransformer(final_config2)
    elements_config, elements_output, _ = tr.parse_blocks2elements()

    computation_config = tr.parse_graph2computation()
    print(tr.elements_config)
    print(computation_config)

