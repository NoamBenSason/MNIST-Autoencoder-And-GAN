from torch.utils.data import DataLoader
from AutoEncoder import Encoder
from AutoEncoder import Decoder
from constants import *
import torchvision.datasets as datasets
import torchvision
from torch import nn

mnist_train_set = datasets.MNIST(root='./data', train=True, download=True,
                                 transform=torchvision.transforms.Compose(
                                     [torchvision.transforms.ToTensor()]))


def main():
    config = {
        'model_name': {'value': 'Encoder'},
        'model_type': {'value': 1},
        "conv_layers_num": 4,
        "fc_layers_num": 4
    }

    for i in range(4):
        config[f"cnn_out_channels_{i}"] = 2
        config[f"cnn_kernel_size_{i}"] = i + 3
        config[f"fc_out_features_{i}"] = 100 - i

    encoder = Encoder((32, 32), 10, config)
    decoder = Decoder(10, (32, 32), config)

    train_dataloader = DataLoader(mnist_train_set, batch_size=1, shuffle=True)
    padding = nn.ConstantPad2d(2, 0)
    pic, _ = next(iter(train_dataloader))
    pic = padding(pic)
    pic.to(device=device)

    x = encoder(pic)
    out = decoder(x)
    pass


if __name__ == '__main__':
    main()
