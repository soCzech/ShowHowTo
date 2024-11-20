import os
import math
import torch
import importlib


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model_checkpoint(model, ckpt):
    assert os.path.exists(ckpt), f"Error: checkpoint file '{ckpt}' not found!"

    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        for k in list(state_dict.keys()):
            if "framestride_embed" in k:
                new_key = k.replace("framestride_embed", "fps_embedding")
                state_dict[new_key] = state_dict[k]
                del state_dict[k]
    model.load_state_dict(state_dict, strict=True)
    print(f">>> model checkpoint '{ckpt}' loaded.", flush=True)
    return model


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._sum += val * n
        self._count += n

    @property
    def value(self):
        if self._count == 0:
            return 0
        return self._sum / self._count
