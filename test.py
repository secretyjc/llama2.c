
import torch
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint as pprint


def load_model(model_path):
    checkpoint = torch.load(model_path)
    pprint(checkpoint.keys())

    # pprint(checkpoint["optimizer"].keys())
    # pprint(checkpoint["iter_num"])
    # pprint(checkpoint["model_args"])
    # pprint(checkpoint["config"])
    # pprint(checkpoint["best_val_loss"])
    layers = [l for l in checkpoint["model"].keys()]
    layers0 = [l for l in checkpoint["model"].keys() if "layers.0" in l]
    params = sum([np.prod(np.array(t.shape)) for l, t in checkpoint["model"].items()])
    print(params)
    for l, t in layers.items():
        if "layers.0" in l or "layers" not in l:
            print(l, t.shape)

    # print(sum(p.numel() for p in checkpoint["model"].parameters()) / 1024 / 1024)

    iter_num = checkpoint['iter_num']
    config = checkpoint['config']
    val_loss = checkpoint['best_val_loss']
    model_args = checkpoint['model_args']
    pprint(checkpoint["model_state_dict"])
    pprint(model_args)

    # model = Transformer(model_args)
    # model.load_state_dict(checkpoint['model'])
    # model.eval()
    # return model

if __name__ == "__main__":
    model_path = "out110M/ckpt.pt"
    # model_path = "out15M/stories15M.pt"
    load_model(model_path)

