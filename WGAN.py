from torch import nn
from constants import device


class Generator(nn.Module):
    def __init__(self, normal_dim, latent_dim, config):
        """
        :param normal_dim: input dimension
        :param latent_dim: output dimension
        :param config:
        """
        super(Generator, self).__init__()
        self.fc_layers_num = config["g_fc_layers_num"]
        self.batch_norm = config["batch_norm"]
        activation_dict = {0: nn.ReLU(), 1: nn.Sigmoid(), 2: nn.Softmax(dim=1),
                           3: nn.LeakyReLU(config["g_relu_slope"])}

        self.activation = activation_dict[config["g_activation"]]

        # ________________________________________________FC INIT:____________________________________________
        self.fc_layers = []

        in_features = normal_dim

        for i in range(self.fc_layers_num - 1):
            curr_layer = nn.Linear(in_features, config[f"g_fc_out_features_{i}"], device=device)
            self.fc_layers.append(curr_layer)
            in_features = config[f"g_fc_out_features_{i}"]

        self.fc_layers.append(nn.Linear(in_features, latent_dim, device=device))  # adding last layer
        self.fc_layers = nn.ModuleList(self.fc_layers)

        # ________________________________________BATCH NORM INIT:____________________________________________

        if self.batch_norm:
            self.batch_norm_layers = []
            for i in range(self.fc_layers_num - 1):
                curr_layer = nn.BatchNorm1d(config[f"g_fc_out_features_{i}"])
                self.batch_norm_layers.append(curr_layer)
            self.batch_norm_layers = nn.ModuleList(self.batch_norm_layers)

    def forward(self, x):
        for i in range(self.fc_layers_num):
            x = self.fc_layers[i](x)
            if self.batch_norm and i != self.fc_layers_num - 1:
                x = self.batch_norm_layers[i](x)
            x = self.activation(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_dim, config):
        """
        :param latent_dim: input dimension
        :param config:
        """
        super(Discriminator, self).__init__()
        self.fc_layers_num = config["d_fc_layers_num"]

        activation_dict = {0: nn.ReLU(), 1: nn.Sigmoid(), 2: nn.Softmax(dim=1),
                           3: nn.LeakyReLU(config["d_relu_slope"])}

        self.activation = activation_dict[config["d_activation"]]

        # ________________________________________________FC INIT:____________________________________________
        self.fc_layers = []
        in_features = latent_dim

        for i in range(self.fc_layers_num - 1):
            curr_layer = nn.Linear(in_features, config[f"d_fc_out_features_{i}"], device=device)
            self.fc_layers.append(curr_layer)
            in_features = config[f"d_fc_out_features_{i}"]

        self.fc_layers.append(nn.Linear(in_features, 1, device=device))  # adding last layer
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, x):
        for i in range(self.fc_layers_num):
            x = self.fc_layers[i](x)
            x = self.activation(x)
        return x
