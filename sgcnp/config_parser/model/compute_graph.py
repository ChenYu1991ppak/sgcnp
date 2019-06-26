

class ComputeInfo(object):
    def __init__(self, compute_cfg):
        self.__dict__.update({
            "task_idx": compute_cfg["task_idx"],
            "train_type": compute_cfg["train_type"],
            "inner": compute_cfg["train_inner"],
            "optim_cfg": compute_cfg["optimizer_cfg"],
            "lr_decay_coef": compute_cfg["lr_decay_coef"],
            "lr_decay_gap": compute_cfg["lr_decay_gap"],
            "epoch": compute_cfg["epoch"],
            "save_dir": compute_cfg["save_dir"],
            "selected_loss": compute_cfg["selected_loss"],
            "extra_output": None
        })


if __name__ == "__main__":
    pass
