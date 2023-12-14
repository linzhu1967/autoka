import torch
import numpy as np
import random
import os

from losses.pairwise import InBatchPairwiseNLL



def normalize(tensor, eps=1e-9):
    """normalize input tensor on last dimension
    """
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)


def set_seed(seed):
    """see: https://twitter.com/chaitjo/status/1394936019506532353/photo/1
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def set_seed_from_config(config):
    if "random_seed" in config:
        random_seed = config["random_seed"]
    else:
        random_seed = 123
    set_seed(random_seed)
    return random_seed


def get_loss(config):
    # if config["loss"] == "PairwiseNLL":
    #     loss = PairwiseNLL()
    # elif config["loss"] == "DistilMarginMSE":
    #     loss = DistilMarginMSE()
    # elif config["loss"] == "KlDiv":
    #     loss = DistilKLLoss()
    # elif config["loss"] == "BCE":
    #     loss = BCEWithLogitsLoss()
    if config["loss"] == "InBatchPairwiseNLL":
        loss = InBatchPairwiseNLL()
    elif config["loss"] == "CrossEntropyLoss":
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("provide valid loss")
    return loss


def rename_keys(d, prefix):
    return {prefix + "_" + k: v for k, v in d.items()}


def restore_model(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
    # strict = False => it means that we just load the parameters of layers which are present in both and
    # ignores the rest
    if len(missing_keys) > 0:
        print("~~ [WARNING] MISSING KEYS WHILE RESTORING THE MODEL ~~")
        print(missing_keys)
    if len(unexpected_keys) > 0:
        print("~~ [WARNING] UNEXPECTED KEYS WHILE RESTORING THE MODEL ~~")
        print(unexpected_keys)
    print("restoring model:", model.__class__.__name__)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def remove_old_ckpt(dir_, k):
    ckpt_names = os.listdir(dir_)
    if len(ckpt_names) <= k:
        pass
    else:
        ckpt_names.remove("model_last.tar")
        if "model_final_checkpoint.tar" in ckpt_names:
            ckpt_names.remove("model_final_checkpoint.tar")
        # TODO
        steps = []
        for ckpt_name in ckpt_names:
            steps.append(int(ckpt_name.split(".")[0].split("_")[-1]))
        oldest = sorted(steps)[0]
        print("REMOVE", os.path.join(dir_, "model_ckpt_{}.tar".format(oldest)))
        os.remove(os.path.join(dir_, "model_ckpt_{}.tar".format(oldest)))


def parse(d, name):
    return {k.replace(name + "_", ""): v for k, v in d.items() if name in k}


