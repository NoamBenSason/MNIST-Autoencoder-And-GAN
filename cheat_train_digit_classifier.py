from sklearn.metrics import balanced_accuracy_score, accuracy_score

import wandb

from constants import *
from torch.utils.data import random_split, DataLoader
import torchvision.datasets as datasets
import torchvision

from ex3.AutoEncoder import Encoder
from ex3.config_loader import load_config
from digit_classifier import DigitClassifier

train_dataloader = None
test_dataloader = None
reload_model = False


def get_config(max_fc_layer_num):
    sweep_config = {}
    sweep_config['method'] = 'bayes'
    sweep_config['metric'] = {'name': 'balanced_accuracy', 'goal': 'maximize'}

    sweep_config['name'] = f"DigitClassifier_{get_time()}"
    param_dict = {
        'model_name': {'value': 'DigitClassifier'},
        'model_type': {'value': 2},
        "fc_layers_num": {'distribution': 'int_uniform', 'min': 1, 'max': max_fc_layer_num},
        'grad': {'value': 0},
        'optimizer': {'values': [0, 1]},
        'activation': {'values': [0, 1, 2]},
        'encoder_index': {'values': list(range(len(models)))}
    }

    for i in range(max_fc_layer_num):
        param_dict[f"fc_out_features_{i}"] = {'distribution': 'int_uniform', 'min': 3, 'max': 25}

    sweep_config['parameters'] = param_dict
    sweep_config['parameters'].update(
        {
            "learning_rate": {'distribution': 'uniform', 'min': 0.0001, 'max': 0.05},
            # "learning_rate": {'values': [0.0001, 0.001, 0.01, 0.1, 0.005, 0.0005, 0.05]},
            'epoch_num': {'distribution': 'int_uniform', 'min': 3, 'max': 30}, })
    # 'epoch_num': {'value': 700}})
    return sweep_config


def collect_batch(batch):
    def one_hot(num: int):
        out = tr.zeros(10)
        out[num] = 1
        return out

    pictures = tr.stack([padding(item[0]) for item in batch])
    labels = tr.stack([one_hot(item[1]) for item in batch])
    return pictures, labels


def load_data(train_batch_size, test_batch_size):
    mnist_train_set = datasets.MNIST(root='./data', train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [torchvision.transforms.ToTensor()]))

    # mnist_train_set, _ = tr.utils.data.random_split(mnist_train_set, [100, len(mnist_train_set) - 100])
    mnist_test_set = datasets.MNIST(root='./data', train=False, download=True,
                                    transform=torchvision.transforms.Compose(
                                        [torchvision.transforms.ToTensor()]))

    train_dataloader1 = DataLoader(mnist_train_set, batch_size=train_batch_size, shuffle=True,
                                   collate_fn=collect_batch)
    test_dataloader1 = DataLoader(mnist_test_set, batch_size=test_batch_size, shuffle=True,
                                  collate_fn=collect_batch)

    return train_dataloader1, test_dataloader1


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        model_path, config_path = models[config["encoder_index"]]
        AE_loaded = tr.load(model_path)
        encoder_config = load_config(config_path)
        custom_encoder = Encoder((32, 32), encoder_config["latent_dim"], encoder_config)
        custom_encoder.load_state_dict(AE_loaded['encoder'])
        wandb.log({"latent_dim": encoder_config["latent_dim"]})

        digit_classifier = DigitClassifier(custom_encoder, encoder_config["latent_dim"], config)
        wandb.watch(digit_classifier)

        criterion = nn.CrossEntropyLoss()
        optimizer_dict = {0: tr.optim.SGD, 1: tr.optim.Adam}
        optimizer = optimizer_dict[config["optimizer"]](digit_classifier.parameters(), lr=config[
            'learning_rate'])
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

                output = digit_classifier(pictures)

                loss = criterion(output, labels)

                if not test_iter:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if test_iter:
                    test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
                else:
                    train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss

                if test_iter:
                    y_pred = tr.argmax(output, dim=1).cpu().detach().numpy()
                    y_true = tr.argmax(labels, dim=1).cpu().detach().numpy()
                    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
                    accuracy = accuracy_score(y_true, y_pred, normalize=True)
                    wandb.log({"epoch": epoch,
                               "test_loss": test_loss,
                               "train_loss": train_loss,
                               "accuracy": accuracy,
                               "balanced_accuracy": balanced_accuracy},
                              step=all_itr)

                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Step [{itr + 1}/{len(train_dataloader)}], "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Test Loss: {test_loss:.4f}"
                    )

                    # saving the model
                    if not reload_model:
                        tr.save({"encoder": custom_encoder.state_dict(),
                                 "digit_classifier": digit_classifier.state_dict()},
                                f"models/cheat_digit_classifier"
                                f"/{config._settings.run_id}_digit_classifier.pth")
                    # if itr > len(train_dataloader) // 2 and test_loss > 0.05:
                    #     wandb.finish()


def main():
    global train_dataloader
    global test_dataloader
    train_dataloader, test_dataloader = load_data(32, 200)
    sweep_id = wandb.sweep(get_config(3), project="ex3_digit_classification_cheat1", entity="malik-noam-idl")
    wandb.agent(sweep_id, train, count=1000)


if __name__ == '__main__':
    main()
