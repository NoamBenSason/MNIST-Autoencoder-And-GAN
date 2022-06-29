from datetime import datetime

import torch as tr
from torch import nn

# device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
device = tr.device("cpu")
padding = nn.ConstantPad2d(2, 0)

npy = lambda x: x.cpu().detach().numpy()

models = [

    # !!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT CHANGE ORDER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # divine sweep - latent dim 5:
    ("models/autoencoder/i4jpcmml_autoencoder.pth", "jsons/divine_sweep.json"),
    # sweepy sweep - latent dim 10:
    ("models/autoencoder/l5b1hdx9_autoencoder.pth", "jsons/sweepy_sweep.json"),
    # dutiful sweep - latent dim 11:
    ("models/autoencoder/ah9ii2vo_autoencoder.pth", "jsons/dutiful_sweep.json"),
    # dauntless sweep - latent dim 16:
    ("models/autoencoder/lrm6q89u_autoencoder.pth", "jsons/dauntless_sweep.json"),
    # misunderstood sweep - latent dim 25:
    ("models/autoencoder/3l7xyv85_autoencoder.pth", "jsons/misunderstood_sweep.json"),
    # !!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT CHANGE ORDER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
]

models2 = [

    # divine sweep - latent dim 5:
    ("models/autoencoder/i4jpcmml_autoencoder.pth", "jsons/divine_sweep.json"),
    # sweepy sweep - latent dim 8:
    ("models/autoencoder/yzoezmge_autoencoder.pth", "jsons/prime_sweep.json"),
    # dutiful sweep - latent dim 11:
    ("models/autoencoder/ah9ii2vo_autoencoder.pth", "jsons/dutiful_sweep.json"),
    # dauntless sweep - latent dim 16:
    ("models/autoencoder/lrm6q89u_autoencoder.pth", "jsons/dauntless_sweep.json"),
    # misunderstood sweep - latent dim 25:
    ("models/autoencoder/3l7xyv85_autoencoder.pth", "jsons/misunderstood_sweep.json"),
]


def one_hot(num: int):
    out = tr.zeros(10)
    out[num] = 1
    return out


one_hot_y_fakes = tr.stack([one_hot(i) for i in range(10) for j in range(10)])


def get_time():
    now = datetime.now()
    return now.strftime("%d-%m-%Y__%H-%M-%S")
