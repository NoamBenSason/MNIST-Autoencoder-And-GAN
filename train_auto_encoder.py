import wandb
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torch.utils.data import DataLoader

from constants import *
import torchvision.datasets as datasets
import torchvision
from AutoEncoder import Encoder, Decoder
from torch import nn
from itertools import chain

train_dataloader = None
test_dataloader = None
reload_model = False


def get_config(max_conv_layer_num, max_fc_layer_num):
    sweep_config = {}
    sweep_config['method'] = 'random'
    sweep_config['metric'] = {'name': 'test_loss', 'goal': 'minimize'}


    sweep_config['name'] = f"AE_{get_time()}"
    param_dict = {
        'model_name': {'value': 'AE'},
        'model_type': {'value': 1},
        "conv_layers_num": {'distribution': 'int_uniform', 'min': 1, 'max': max_conv_layer_num},
        "fc_layers_num": {'distribution': 'int_uniform', 'min': 1, 'max': max_fc_layer_num},
        "latent_dim": {'distribution': 'int_uniform', 'min': 9, 'max': 16}

    }

    for i in range(max_conv_layer_num):
        param_dict[f"cnn_out_channels_{i}"] = {'distribution': 'int_uniform', 'min': 20,
                                               'max': 30}
        param_dict[f"cnn_kernel_size_{i}"] = {'distribution': 'int_uniform', 'min': 2, 'max': 4}

    for i in range(max_fc_layer_num):
        param_dict[f"fc_out_features_{i}"] = {'distribution': 'int_uniform',
                                              'min': 20 * (max_fc_layer_num - i),
                                              'max': 70 * (max_fc_layer_num - i)}


    sweep_config['parameters'] = param_dict
    sweep_config['parameters'].update(
        {"learning_rate": {'distribution': 'uniform', 'min': 0.0005, 'max': 0.005},
         'epoch_num': {'distribution': 'int_uniform', 'min': 3, 'max': 7}})
    return sweep_config


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        latent_dim = config['latent_dim']
        encoder_model = Encoder((32, 32), latent_dim, config)
        decoder_model = Decoder(latent_dim, (32, 32), config)
        wandb.watch(encoder_model)
        wandb.watch(decoder_model)

        criterion = nn.MSELoss()
        optimizer = tr.optim.Adam(chain(encoder_model.parameters(), decoder_model.parameters()),
                                  lr=config['learning_rate'])
        train_loss = 1.0
        test_loss = 1.0
        test_interval = 50

        all_itr = 0
        num_epochs = config['epoch_num']
        for epoch in range(num_epochs):
            itr = 0
            for pictures, labels in train_dataloader:  # getting training batches
                itr += 1
                all_itr += 1
                if (itr + 1) % test_interval == 0:
                    test_iter = True
                    pictures, labels = next(iter(test_dataloader))
                else:
                    test_iter = False
                # _______________________________________TRAINING:____________________________________________

                latent_output = encoder_model(pictures)
                output = decoder_model(latent_output)

                loss = criterion(pictures, output)

                if not test_iter:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if test_iter:
                    test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
                else:
                    train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss

                if test_iter:
                    wandb.log({"epoch": epoch,
                               "test_loss": test_loss,
                               "train_loss": train_loss},
                              step=all_itr)
                    # ------------------------------------------------------
                    #                   LOGGING PICTURES
                    # ------------------------------------------------------

                    picture_out1, picture_out2, picture_out3 = output[0], output[len(output) // 2], output[-1]
                    picture_org1, picture_org2, picture_org3 = pictures[0], pictures[len(output) // 2], \
                                                               pictures[-1]
                    grid = torchvision.utils.make_grid([picture_out1, picture_out2, picture_out3,
                                                        picture_org1, picture_org2, picture_org3], nrow=3)
                    images = wandb.Image(grid.cpu().detach(), caption="Top: Output, Bottom: Input")
                    wandb.log({"examples": images}, step=all_itr)

                    # ------------------------------------------------------
                    #                   /LOGGING PICTURES
                    # ------------------------------------------------------

                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Step [{itr + 1}/{len(train_dataloader)}], "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Test Loss: {test_loss:.4f}"
                    )

                    # saving the model
                    if not reload_model:
                        tr.save({"encoder": encoder_model.state_dict(),
                                 "decoder": decoder_model.state_dict()},
                                f"models/{config._settings.run_id}_autoencoder.pth")
                    if itr > len(train_dataloader) // 2 and test_loss > 0.05:
                        wandb.finish()


def load_data(train_batch_size: int, test_batch_size: int):
    global train_dataloader
    global test_dataloader
    mnist_train_set = datasets.MNIST(root='./data', train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [torchvision.transforms.ToTensor()]))
    mnist_test_set = datasets.MNIST(root='./data', train=False, download=True,
                                    transform=torchvision.transforms.Compose(
                                        [torchvision.transforms.ToTensor()]))

    train_dataloader = DataLoader(mnist_train_set, batch_size=train_batch_size, shuffle=True,
                                  collate_fn=collect_batch)
    test_dataloader = DataLoader(mnist_test_set, batch_size=test_batch_size, shuffle=True,
                                 collate_fn=collect_batch)

    return train_dataloader, test_dataloader


def collect_batch(batch):
    pictures = tr.stack([padding(item[0]) for item in batch])
    labels = tr.FloatTensor([item[1] for item in batch])
    return pictures, labels


def main():
    load_data(32, 32)
    sweep_id = wandb.sweep(get_config(2, 5), project="ex3_autoencoder1",
                           entity="malik-noam-idl")
    wandb.agent(sweep_id, train, count=30)


if __name__ == '__main__':
    main()
