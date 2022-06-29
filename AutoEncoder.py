from typing import Tuple, Sequence
from constants import *
import torch as tr
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_dims: Sequence[int,], latent_dim: int, config):
        super(Encoder, self).__init__()
        self.conv_layers_num = config["conv_layers_num"]
        self.fc_layers_num = config["fc_layers_num"]
        self.relu = nn.ReLU()

        # ________________________________________________CNN INIT____________________________________________
        self.conv_layers = []

        next_in_channels = 1
        H, W = input_dims
        for i in range(self.conv_layers_num):
            curr_layer = nn.Conv2d(next_in_channels, config[f"cnn_out_channels_{i}"],
                                   config[f"cnn_kernel_size_{i}"],
                                   device=device)
            self.conv_layers.append(curr_layer)
            next_in_channels = config[f"cnn_out_channels_{i}"]

            H = H - (config[f"cnn_kernel_size_{i}"] - 1)
            W = W - (config[f"cnn_kernel_size_{i}"] - 1)

        config["cnn_final_out_dim"] = next_in_channels
        self.conv_layers = nn.ModuleList(self.conv_layers)
        # ________________________________________________FC INIT_____________________________________________
        self.fc_layers = []
        fc_in_features = H * W * next_in_channels
        config["cnn_dim"] = fc_in_features
        for i in range(self.fc_layers_num - 1):
            curr_layer = nn.Linear(fc_in_features, config[f"fc_out_features_{i}"], device=device)
            self.fc_layers.append(curr_layer)
            fc_in_features = config[f"fc_out_features_{i}"]

        self.fc_layers.append(nn.Linear(fc_in_features, latent_dim, device=device))  # adding last layer
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, x):

        for i in range(self.conv_layers_num):
            x = self.conv_layers[i](x)
            x = self.relu(x)

        x = tr.reshape(x, [x.shape[0], -1])

        for i in range(self.fc_layers_num - 1):
            x = self.fc_layers[i](x)
            x = self.relu(x)

        x = self.fc_layers[self.fc_layers_num - 1](x)  # Without activation
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dims: Sequence[int,], config):
        super(Decoder, self).__init__()
        self.trans_conv_layers_num = config["conv_layers_num"]
        self.fc_layers_num = config["fc_layers_num"]
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.latent_dim = latent_dim

        # ________________________________________________FC INIT:____________________________________________
        self.fc_layers = []
        fc_in_features = latent_dim

        for i in reversed(range(self.fc_layers_num - 1)):
            curr_layer = nn.Linear(fc_in_features, config[f"fc_out_features_{i}"], device=device)
            self.fc_layers.append(curr_layer)
            fc_in_features = config[f"fc_out_features_{i}"]

        cnn_dim = config["cnn_dim"]
        self.fc_layers.append(nn.Linear(fc_in_features, cnn_dim, device=device))  # adding last layer
        self.fc_layers = nn.ModuleList(self.fc_layers)

        # ________________________________________________CNN INIT:___________________________________________
        self.trans_conv_layers = []

        next_in_channels = config["cnn_final_out_dim"]
        self.H, self.W = int((cnn_dim / next_in_channels) ** 0.5), int((cnn_dim / next_in_channels) ** 0.5)

        for i in reversed(range(1, self.trans_conv_layers_num)):
            curr_layer = nn.ConvTranspose2d(next_in_channels, config[f"cnn_out_channels_{i}"],
                                            config[f"cnn_kernel_size_{i}"],
                                            device=device)
            self.trans_conv_layers.append(curr_layer)
            next_in_channels = config[f"cnn_out_channels_{i}"]

        self.trans_conv_layers.append(nn.ConvTranspose2d(next_in_channels, 1,
                                                         config[f"cnn_kernel_size_{0}"],
                                                         device=device))
        self.trans_conv_layers = nn.ModuleList(self.trans_conv_layers)

    def forward(self, x):

        for i in range(self.fc_layers_num - 1):
            x = self.fc_layers[i](x)
            x = self.relu(x)

        x = self.fc_layers[self.fc_layers_num - 1](x)
        x = tr.reshape(x, [x.shape[0], -1, self.H, self.W])

        for i in range(self.trans_conv_layers_num - 1):
            x = self.trans_conv_layers[i](x)
            x = self.relu(x)

        x = self.trans_conv_layers[self.trans_conv_layers_num - 1](x)
        x = self.sigmoid(x)
        return x
