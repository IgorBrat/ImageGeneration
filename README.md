# Image Generation Project

Image generation has become really popular these days with growing community, open-source researches and technical
solutions capable of running generative models. That's why the main idea of the project is to generate images with
controllable features (for example, Michael Jackson with a cowboy hat in Babylon - quite exciting, right?)

First models developed for image generation were generative adversial networks (GANs). As generated images should not
copy the real ones but rather generate similar in a **perceptual way**. For that new model architecture was introduced:
except of training one model to generate samples, it was advised to use two models approach - generator and
discriminator (**G** and **D** later in the document). While D is responsible to distinguish real from fake (generated)
images, G is trying to fool D and produce as real looking images as possible. That way, G and D are continiously
training and, unlike traditional computer vision models, can never stop learning. That happens because both models are
learning from each other: if D is becoming more correct in guessing real and fake images, G is becoming a greater '
artist' to fool D and generate more realistic samples. However, there can be a problem of one model becoming more
superior that the other, thus stops actually learning, and the worse one can't catch up with the better. So training
models in equal conditions is advised. More complex solutions are not described or used in the project.

## General info

Models, utils and models resources are in `ml\` directory. More on each model can be read in relevant docstrings.\
Web application is in `web\` directory.\
Tests for the models are implemented in `tests\` directory.

There are some resources that are not in the repo:

1. Datasets should be in `datasets\` directory. Advised
   datasets: [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/),
   [CELEBa](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html),
   [dog breeds](http://vision.stanford.edu/aditya86/ImageNetDogs/),
   [anime faces](https://www.kaggle.com/datasets/splcher/animefacedataset).
2. Trained models weights should be stored in `ml\resources\` directory in files with .pt or .pth format. Weights for
3. Inception and VGG networks, used in style transfer and metrics, should also be stored in `ml\resources\`.

## Models

### Basic Generative models

Basic generative models include modifications of basic GAN, and also a similar in its architecture Variational
Auto-Encoder. All these models are generally used to create low-resolution images because of their relatively simple
architectures.\
Project includes these models:

- Deep Convolutional GAN
- Conditional GAN
- Controllable GAN
- Style GAN
- Variational Auto-Encoder

More about the usage of the models is listed in section **Web**.

### Text2Image Generative models

As GANs and their competitors were introduced to sample random images from learned distribution, there appeared a
problem of controllable generation, especially by using text prompts. So Text2Image generative models were created: they
have more complex architectures than simple GANs, though very similar to them, and include **Text encoding** model,
which
encodes given query to then be fed to generative model.\
Project includes:

- Text2Image DCGAN
- StackGAN

## Style Transfer

As an additional feature, Style Transfer is implemented in the project. It is used to apply style from some source
image (advised to use popular art - Monet, Da Vinci, Pollock etc.; experiment with it!☺) to destination image. It
generally includes background colors, color gamma. Neural style transfer is used in the project.
More on that topic [here](https://arxiv.org/pdf/1508.06576.pdf).

## Web

Web application is implemented using Django. It can be started on 'localhost:8000' from root directory by
running `py manage.py runserver`. To change production port and other settings check the official
[documentation](https://docs.djangoproject.com/en/5.0/).
Application has 5 main pages:

- Home: general information about the project;
- Train: training different models with hyperparameter adjusting;
- Inference: generate samples from trained models, each stored in `ml\resources\<model_type>\<model_version>`;
- Style Transfer: apply style from image to another with adjustant hyperparameters;
- Logs: information about training each model, including errors and training time. Logs are stored
  in `ml\resources\logs` directory.

### Train

First choose the model you want to train. After that you are redirected to a page with a choice of different
initialisation and hyperparameters, and datasets as well, to train the model. For now the model will be run in thread:
it can cause potential problems with long-running processes, so **should be avoided in production**! Information about
the model can be dynamically viewed on Logs page in relevant model log file: `<model_type-start_datetime>.txt`\
***For now training pages and logging only for DCGAN, Text2Image DCGAN and StackGAN are implemented.***

Models are not being saved at development phase. If you want to save model after (or during) training, add
`save()` method call in `train()` method. Logging also should be implemented in that method.

Text2Image DCGAN and StackGAN support only CUB dataset (or other datasets yielding text attributes).

### Inference

You can choose the model and its version to create samples of it and thus check its results perceptually.
Inference is different for some models, where you should provide text prompt for Text2Image models or desired
class-feature (for now only class indices are supported).

## Tests

Unit tests can be run from root directory by running `py -m pytest`.
As for now, the tests for models, utils (except metrics) and style transfer are implemented.

## Metrics

After training models, they can be compared using two metrics, implemented in the project: Fréchet Inception Distance (
FID) and Perceptual Path Length (PPL).

- Fréchet distance finds the shortest distance needed to walk along two lines, or two curves, simultaneously.
  The most intuitive explanation of Fréchet distance is as the "minimum leash distance" between two points.
  Lower FID means that the model is generating realistic images, as the "distance" between generated and real ones is
  shorter.
- PPL measures the expected "jarringness" of each step in the interpolation between two images when you add together the
  jarringness of steps from random paths.
  Lower PPL is better.

## Future plans

1. For now, trained models weights are stored in `ml\resources\<model_type>\<model_version>\` directories, which does
   not give any information on the model itself. It is advised to name directory properly instead of '<model_version>\',
   or store text file with information about model.
2. Models are being trained in threads, which is not the way Django actually works. So in long-running trainings or
   during production it's not advised to use that approach.