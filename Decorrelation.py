import torch as tr
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import DataLoader
from AutoEncoder import Encoder, Decoder
from config_loader import load_config
from constants import *
import numpy as np
import matplotlib.pyplot as plt
import itertools

import seaborn as sns

import os




def collect_batch(batch):
    pictures = tr.stack([padding(item[0]) for item in batch])
    labels = tr.FloatTensor([item[1] for item in batch])
    return pictures, labels


def load_data(samples_num) -> tr.FloatTensor:
    mnist_set = datasets.MNIST(root='./data', train=True, download=True,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor()]))

    dataloader = DataLoader(mnist_set, batch_size=samples_num, shuffle=True,
                            collate_fn=collect_batch)

    return next(iter(dataloader))[0]


def plot(plot_dict):
    items = sorted(list(plot_dict.items()), key=lambda x: x[0])
    dims = [item[0] for item in items]
    values = [item[1] for item in items]

    plt.title(f"Correlation between coordinates")
    plt.plot(dims, values, label="Max_Correlation", color="#15b38b", marker="o")
    plt.xlabel("Latent dimension")
    plt.ylabel("Averaged correlation")
    plt.xticks(dims, [str(dim) for dim in dims])
    plt.legend()
    plt.savefig("averaged_correlation/" + get_time() + ".png")
    plt.show()


def reduce_to_single_value(corr_matrix):
    abs_values = [abs(corr_matrix[i][j].item()) for i, j in itertools.combinations(
        range(corr_matrix.shape[0]), 2)]
    # return sum(abs_values) / len(abs_values)
    return max(abs_values)


def plot_heat_map(corr_matrix):
    corr_matrix = corr_matrix.cpu().detach().numpy()
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(corr_matrix, mask=mask, vmax=1.0, vmin=-1.0, square=True, cmap="PiYG", center=0)
        plt.title(f"Correlation between coordinates with latent dimension {str(corr_matrix.shape[0])}")
        plt.show()


def main():
    plot_dict = {}
    for model_path, config_path in models2:
        samples = load_data(15000)
        AE_loaded = tr.load(model_path)
        config = load_config(config_path)
        encoder = Encoder((32, 32), config["latent_dim"], config)
        encoder.load_state_dict(AE_loaded['encoder'])

        x = encoder(samples)
        corr_matrix = tr.corrcoef(x.T)
        plot_heat_map(corr_matrix)
        corr = reduce_to_single_value(corr_matrix)
        plot_dict[config["latent_dim"]] = corr

    plot(plot_dict)


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
