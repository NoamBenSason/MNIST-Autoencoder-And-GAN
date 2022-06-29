from torch import nn
from constants import *


class DigitClassifier(nn.Module):

    def __init__(self, encoder, latent_dim, config):
        super(DigitClassifier, self).__init__()

        self.fc_layers_num = config["fc_layers_num"]
        activation_dict = {0: nn.ReLU(), 1: nn.Sigmoid(), 2: nn.Softmax(dim=1)}

        self.activation = activation_dict[config["activation"]]
        self.encoder = encoder

        if config["grad"]:
            self.add_module("encoder", self.encoder)
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
        # ________________________________________________FC INIT_____________________________________________
        self.fc_layers = []
        fc_in_features = latent_dim
        for i in range(self.fc_layers_num - 1):
            curr_layer = nn.Linear(fc_in_features, config[f"fc_out_features_{i}"], device=device)
            self.fc_layers.append(curr_layer)
            fc_in_features = config[f"fc_out_features_{i}"]

        self.fc_layers.append(nn.Linear(fc_in_features, 10, device=device))  # adding last layer
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def forward(self, x):
        x = self.encoder(x)

        for i in range(self.fc_layers_num):
            x = self.fc_layers[i](x)
            x = self.activation(x)
        return x
