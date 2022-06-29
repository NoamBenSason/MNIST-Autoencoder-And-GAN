import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from itertools import combinations

import wandb
import torch as tr
from constants import get_time, padding, models, one_hot_y_fakes, npy
from ex3.AutoEncoder import Encoder, Decoder
from ex3.config_loader import load_config
from CWGAN import Generator, Discriminator
from digit_classifier import DigitClassifier

train_dataloader = None
reload_model = False


def get_config(max_g_fc_layer_num, max_d_fc_layer_num):
    sweep_config = {}
    sweep_config['method'] = 'bayes'
    sweep_config['metric'] = {'name': 'expected_loss', 'goal': 'minimizea'}

    sweep_config['name'] = f"CWGAN_{get_time()}"
    param_dict = {
        'model_name': {'value': 'CWGAN'},
        'model_type': {'value': 4},
        "g_fc_layers_num": {'distribution': 'int_uniform', 'min': 1, 'max': max_g_fc_layer_num},
        "d_fc_layers_num": {'distribution': 'int_uniform', 'min': 1, 'max': max_d_fc_layer_num},
        "batch_norm": {'values': [0]},
        'g_activation': {'values': [3]},
        'd_activation': {'values': [3]},
        'encoder_index': {'values': list(range(1, len(models)))},
        'g_relu_slope': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
        'd_relu_slope': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
        'normal_dim': {'distribution': 'int_uniform', 'min': 10, 'max': 120},
        # 'n_critic': {'distribution': 'int_uniform', 'min': 1, 'max': 1},
        'd_n_critic': {'values': [1]},
        'g_n_critic': {'values': [1]},
        'clip': {'distribution': 'uniform', 'min': 0.05, 'max': 0.15},
        'embedding_dim': {'distribution': 'int_uniform', 'min': 30, 'max': 150}
    }

    for i in range(max_g_fc_layer_num):
        param_dict[f"g_fc_out_features_{i}"] = {'distribution': 'int_uniform', 'min': 15, 'max': 100}

    for i in range(max_d_fc_layer_num):
        param_dict[f"d_fc_out_features_{i}"] = {'distribution': 'int_uniform', 'min': 15, 'max': 100}

    sweep_config['parameters'] = param_dict
    sweep_config['parameters'].update(
        {"g_learning_rate": {'distribution': 'uniform', 'min': 0.01, 'max': 0.045},
         "d_learning_rate": {'distribution': 'uniform', 'min': 0.0001, 'max': 0.001},
         'epoch_num': {'distribution': 'int_uniform', 'min': 4, 'max': 10}})
    return sweep_config


def collect_batch(batch):
    pictures = tr.stack([padding(item[0]) for item in batch])
    # labels = tr.stack([one_hot(item[1]) for item in batch])
    labels = tr.IntTensor([item[1] for item in batch])
    return pictures, labels


def load_data(train_batch_size):
    mnist_train_set = datasets.MNIST(root='./data', train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [torchvision.transforms.ToTensor()]))

    train_dataloader1 = DataLoader(mnist_train_set, batch_size=train_batch_size, shuffle=True,
                                   collate_fn=collect_batch)

    return train_dataloader1


def sample_normal_uniform(batch_size, normal_dim):
    return tr.FloatTensor(np.random.normal(0, 1, (batch_size, normal_dim)))


def sample_fake_labels(batch_size, embed_func):
    fake_scalars = tr.from_numpy(np.random.randint(0, 9, batch_size))
    # return tr.stack([one_hot(item) for item in fake_scalars])
    return embed_func(fake_scalars)


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        model_path, config_path = models[config["encoder_index"]]
        AE_loaded = tr.load(model_path)
        AE_config = load_config(config_path)
        custom_encoder = Encoder((32, 32), AE_config["latent_dim"], AE_config)
        custom_encoder.load_state_dict(AE_loaded['encoder'])
        custom_decoder = Decoder(AE_config["latent_dim"], (32, 32), AE_config)
        custom_decoder.load_state_dict(AE_loaded['decoder'])
        wandb.log({"latent_dim": AE_config["latent_dim"]})

        # _____________________________________Statistics Parameters__________________________________________
        softmax_dim1 = tr.nn.Softmax(dim=1)
        ce_loss = tr.nn.CrossEntropyLoss()
        ref_encoder_path, ref_encoder_config_path = models[4]
        ref_encoder_load = tr.load(ref_encoder_path)
        ref_AE_config = load_config(ref_encoder_config_path)
        ref_encoder = Encoder((32, 32), ref_AE_config["latent_dim"], ref_AE_config)
        ref_encoder.load_state_dict(ref_encoder_load['encoder'])

        classifier_path, config_classifier_path = \
            "models/cheat_digit_classifier/3sqrb7l2_digit_classifier.pth", "jsons/classifier_hearty_sweep.json"
        ref_classifier_load = tr.load(classifier_path)
        ref_classifier_config = load_config(config_classifier_path)
        ref_classifier = DigitClassifier(ref_encoder, ref_AE_config["latent_dim"], ref_classifier_config)
        ref_classifier.load_state_dict(ref_classifier_load['digit_classifier'])
        ref_classifier.encoder.load_state_dict(ref_classifier_load['encoder'])

        for param in ref_encoder.parameters():
            param.requires_grad = False
        for param in ref_classifier.parameters():
            param.requires_grad = False
        for param in ref_classifier.encoder.parameters():
            param.requires_grad = False

        # ____________________________________/Statistics Parameters__________________________________________

        for param in custom_encoder.parameters():
            param.requires_grad = False

        for param in custom_decoder.parameters():
            param.requires_grad = False

        embedding = tr.nn.Embedding(10, config['embedding_dim'])

        generator = Generator(config['normal_dim'], AE_config["latent_dim"], config, config['embedding_dim'])
        discriminator = Discriminator(AE_config["latent_dim"], config, config['embedding_dim'])
        wandb.watch(generator)
        wandb.watch(discriminator)

        optimizer_G = tr.optim.RMSprop(generator.parameters(), lr=config['g_learning_rate'])
        optimizer_D = tr.optim.RMSprop(discriminator.parameters(), lr=config['d_learning_rate'])
        clip = config['clip']

        avg_generator_loss = 1.0
        avg_discriminator_loss = 1.0
        log_interval = 100

        y_fakes_to_gen = embedding(tr.IntTensor([i for i in range(10) for j in range(10)]))
        d_n_critic = config['d_n_critic']
        g_n_critic = config['g_n_critic']

        batches_done = 0
        num_epochs = config['epoch_num']
        for epoch in range(num_epochs):
            # temp = d_n_critic
            # d_n_critic = g_n_critic
            # g_n_critic = temp
            for i, (real_pictures, real_labels) in enumerate(train_dataloader):
                real_labels = embedding(real_labels)
                y_fake = sample_fake_labels(real_pictures.shape[0], embedding)
                y_fake_copy = tr.clone(y_fake).detach()

                if (i + 1) % d_n_critic == 0:
                    # Train the discriminator every n_critic iterations
                    # _____________________________________Train Discriminator____________________________________
                    optimizer_D.zero_grad()

                    # Generating random input for the generator
                    z = sample_normal_uniform(real_pictures.shape[0], config["normal_dim"])

                    # Processing the generated input in the generator
                    fake_pictures = generator(z, y_fake).detach()

                    # Discriminator loss
                    real_images_scores = discriminator(custom_encoder(real_pictures), real_labels)
                    fake_images_scores = discriminator(fake_pictures, y_fake)
                    loss_D = -tr.mean(real_images_scores) + tr.mean(fake_images_scores)
                    avg_discriminator_loss = 0.9 * float(loss_D.detach()) + 0.1 * avg_discriminator_loss

                    # Discriminator Backpropagation
                    loss_D.backward()
                    optimizer_D.step()

                    # Clip weights of discriminator
                    # for param in discriminator.parameters():
                    #     param.data.clamp_(-clip, clip)

                if (i + 1) % g_n_critic == 0:
                    # ________________________________________Train Generator_________________________________
                    optimizer_G.zero_grad()

                    # Generating pictures to be trained on
                    z = sample_normal_uniform(real_pictures.shape[0], config["normal_dim"])
                    # y_fake = sample_fake_labels(real_pictures.shape[0], embedding)
                    trained_fake_pictures = generator(z, y_fake_copy)
                    # Generator loss
                    loss_G = -tr.mean(discriminator(trained_fake_pictures, y_fake_copy))
                    avg_generator_loss = 0.9 * float(loss_G.detach()) + 0.1 * avg_generator_loss

                    loss_G.backward()
                    optimizer_G.step()
                batches_done += 1

                # Determining if this is a logging iteration
                log_iter = (i + 1) % log_interval == 0

                if log_iter:
                    wandb.log({"epoch": epoch,
                               "generator_loss": avg_generator_loss,
                               "discriminator_loss": avg_discriminator_loss},
                              step=batches_done)

                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], "
                        f"Step [{i + 1}/{len(train_dataloader)}], "
                        f"Generator Loss: {avg_generator_loss:.4f}, "
                        f"Discriminator Loss: {avg_discriminator_loss:.4f}"
                    )

                    # ________________________________________________________________________________________
                    #                                LOGGING GENERATED PICTURES
                    # ________________________________________________________________________________________
                    generated_images = None
                    with tr.no_grad():
                        generated_images = custom_decoder(
                            generator(sample_normal_uniform(100, config["normal_dim"]), y_fakes_to_gen))

                    grid = torchvision.utils.make_grid(generated_images, nrow=10)
                    images = wandb.Image(grid.cpu().detach(), caption="Generated Images")
                    wandb.log({"generated_images": images}, step=batches_done)

                    # ________________________________________________________________________________________
                    #                               /LOGGING GENERATED PICTURES
                    # ________________________________________________________________________________________

                    # ________________________________________________________________________________________
                    #                        MEASURING METRICS ON GENERATED IMAGES
                    # ________________________________________________________________________________________

                    flat_generated = tr.reshape(generated_images, [generated_images.shape[0], -1])
                    generation_variance = tr.mean(tr.var(flat_generated, dim=0)).item()
                    wandb.log({"generated_images_variance": generation_variance}, step=batches_done)

                    classifications = softmax_dim1(ref_classifier(generated_images))
                    avg_recognition = tr.mean(tr.max(classifications, dim=1).values).item()
                    wandb.log({"average_recognition_value": avg_recognition}, step=batches_done)

                    balanced_performance = 0.08 * avg_recognition + generation_variance
                    wandb.log({"balanced_performance": balanced_performance}, step=batches_done)

                    expected_loss = ce_loss(classifications, one_hot_y_fakes).item()
                    wandb.log({"expected_loss": expected_loss}, step=batches_done)
                    # ________________________________________________________________________________________
                    #                       /MEASURING METRICS ON GENERATED IMAGES
                    # ________________________________________________________________________________________

                    # saving the model
                    if not reload_model:
                        tr.save({"generator": generator.state_dict(),
                                 "discriminator": discriminator.state_dict(), },
                                f"models/CWGAN/{config._settings.run_id}_CWGAN.pth")
        wandb.log({})


def main():
    global train_dataloader
    train_dataloader = load_data(60)
    sweep_id = wandb.sweep(get_config(3, 20), project="ex3_CWGAN_inverted1",
                           entity="malik-noam-idl")
    wandb.agent(sweep_id, train, count=1000)


if __name__ == '__main__':
    main()
