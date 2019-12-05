# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
from random import seed, randint
import config


def generate_random_examples(n, Gs, kind):
    """Generate n random examples from the genenerator Gs
    
    Arguments:
        n {int} -- number of examples to generate
        Gs {Generator} -- model to use
        kind {str} -- Name of subdirectory to store generated images
    """
    seed(7)
    for i in range(n):
        # Pick latent vector.
        rnd = np.random.RandomState(randint(0, 200000))
        latents = rnd.randn(1, Gs.input_shape[1])

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, kind, 'example_{}.png'.format(i))
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

def custom_latent_to_image(latent, Gs, result):
    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latent, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # Save image.
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, result)
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)  

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    def load_gs(url):
        with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
            # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
            # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
            # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
        return Gs
        

    # # Print network details.
    # Gs.print_layers()

    url_faces = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'      # karras2019stylegan-ffhq-1024x1024.pkl
    url_cars = 'https://drive.google.com/uc?id=1MJ6iCfNtMIRicihwRorsM3b7mmtmK9c3'       # karras2019stylegan-cars-512x384.pkl
    url_bedrooms = 'https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF'   # karras2019stylegan-bedrooms-256x256.pkl

    generate_random_examples(10, load_gs(url_faces), 'faces')
    generate_random_examples(10, load_gs(url_cars), 'cars')
    generate_random_examples(10, load_gs(url_bedrooms), 'bedrooms')
    #custom_latent_to_image(np.load('data/latent_representations/our/rosh1_aligned.npy'), load_gs(url_faces), 'custom_face.png')

if __name__ == "__main__":
    main()
