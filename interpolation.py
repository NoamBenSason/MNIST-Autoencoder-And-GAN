import os

import numpy as np
import torch as tr
import torchvision
import torchvision.datasets as datasets

from constants import npy, get_time, padding
from config_loader import load_config
from AutoEncoder import Encoder, Decoder
from WGAN import Generator
from torch.utils.data import DataLoader

gan_path, gan_config = "models/WGAN/7grfls5q_WGAN.pth", "jsons/gan_icy_sweep.json"
AE_path, AE_config_path = "models/autoencoder/3l7xyv85_autoencoder.pth", "jsons/misunderstood_sweep.json"
gan_AE_path, gan_AE_config_path = "models/autoencoder/l5b1hdx9_autoencoder.pth", "jsons/sweepy_sweep.json"


def load_data(train_batch_size: int):
    mnist_train_set = datasets.MNIST(root='./data', train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [torchvision.transforms.ToTensor()]))
    train_dataloader = DataLoader(mnist_train_set, batch_size=train_batch_size, shuffle=True,
                                  collate_fn=collect_batch)
    return train_dataloader


def collect_batch(batch):
    pictures = tr.stack([padding(item[0]) for item in batch])
    return pictures


def generate_random(generator: Generator, decoder: Decoder, normal_dim):
    z = tr.FloatTensor(np.random.normal(0, 1, (10, normal_dim)))
    latent_codes = generator(z).detach()
    pictures = decoder(latent_codes).detach()
    for i in range(len(latent_codes)):
        time = get_time()
        tr.save(latent_codes[i], f"interpolation_data/from_gan/{time}_{i}_code.pt")
        torchvision.utils.save_image(pictures[i], f"interpolation_data/from_gan/{time}_{i}_pic.png")


def encode_random(encoder: Encoder, decoder: Decoder):
    real_pictures = next(iter(load_data(10)))
    latent_codes = encoder(real_pictures).detach()
    pictures = decoder(latent_codes).detach()
    for i in range(len(latent_codes)):
        time = get_time()
        tr.save(latent_codes[i], f"interpolation_data/from_encoder/{time}_{i}_code.pt")
        torchvision.utils.save_image(pictures[i], f"interpolation_data/from_encoder/{time}_{i}_pic.png")


def generate_l1_l2(generator, wgan_decoder, encoder, WGAN_config, AE_decoder):
    generate_random(generator, wgan_decoder, WGAN_config['normal_dim'])
    encode_random(encoder, AE_decoder)


def interpolate(l1, l2, decoder, interpolate_type):
    alpha_num = 20
    assert alpha_num > 1
    alpha_values = [i / (alpha_num - 1) for i in range(alpha_num)]

    interp_lst = [(a * l1 + (1 - a) * l2) for a in alpha_values]
    decoded = decoder(tr.stack(interp_lst)).detach()
    dir_path = f"interpolation_data/interpolations/{get_time()}_{interpolate_type}"
    os.mkdir(dir_path)
    for i, latent in enumerate(decoded):
        torchvision.utils.save_image(decoded[i], f"{dir_path}/{i}_pic.png")

    grid = torchvision.utils.make_grid(decoded, nrow=alpha_num)
    torchvision.utils.save_image(grid, f"{dir_path}/grid.png")


def main():
    AE_loaded = tr.load(AE_path)
    AE_config = load_config(AE_config_path)
    encoder = Encoder((32, 32), AE_config["latent_dim"], AE_config)
    encoder.load_state_dict(AE_loaded['encoder'])
    AE_decoder = Decoder(AE_config["latent_dim"], (32, 32), AE_config)
    AE_decoder.load_state_dict(AE_loaded['decoder'])

    WGAN_loaded = tr.load(gan_path)
    WGAN_config = load_config(gan_config)
    generator = Generator(WGAN_config['normal_dim'], 10, WGAN_config)
    generator.load_state_dict(WGAN_loaded['generator'])

    wgan_AE_loaded = tr.load(gan_AE_path)
    wgan_AE_config = load_config(gan_AE_config_path)
    wgan_decoder = Decoder(10, (32, 32), wgan_AE_config)
    wgan_decoder.load_state_dict(wgan_AE_loaded['decoder'])

    loaded_generator_l1 = tr.load("interpolation_data/from_gan/chosen/10-01-2022__20-52-50_code_1.pt")
    loaded_generator_l2 = tr.load("interpolation_data/from_gan/chosen/10-01-2022__20-52-50_code_6.pt")
    loaded_encoder_l1 = tr.load("interpolation_data/from_encoder/chosen/10-01-2022__21-03-22_0_code.pt")
    loaded_encoder_l2 = tr.load("interpolation_data/from_encoder/chosen/10-01-2022__21-03-22_8_code.pt")

    interpolate(loaded_generator_l1, loaded_generator_l2, wgan_decoder, "generator")
    interpolate(loaded_encoder_l1, loaded_encoder_l2, AE_decoder, "encoder")


if __name__ == '__main__':
    main()
