# Image generation using Variational Autoencoder (VAE)

## About This Project
This project was done in the cnetxt of the first semester course 'Object Oriented Programming with Python'
of my master in Data Science.
Final Grade: ***not yet received***
It demonstrates the implementation of a variational autoencoder using neural networks.
In this project I implemented a Variational Autoencoder (VAE) using TensorFlow.
It includes a modular architecture (Encoder, Decoder, BiCoder), a custom DataLoader, training utilities, 
and scripts for visualizing the latent space using t-SNE. 
The goal of this project is to demonstrate object-oriented code design, probabilistic modeling, and deep learning 
implementation skills for machine learning and data science roles.

## Methods & Concepts used
- Variational inference  
- Bayesian latent variable models  
- KL divergence minimization  
- Reparameterization trick  
- Object-oriented programming in Python  
- t-SNE dimensionality reduction
  
## Project Structure
├── train_vae.py # Main training script with argparse interface
├── vae.py # VAE model, Encoder, Decoder, training logic
├── dataloader.py # Loads MNIST (color or bw), batching, normalization
├── utils.py # Helper functions (saving images etc.)
├── argparser.py # Argument parser with dataset/epoch/output options
└── losses.py # KL divergence and log-likelihood implementations

## Files
bw_data.zip contains:
- mnist_bw.npy -> black and white training images
- mnist_bw_te.npy -> black and white test images
- mnist_bw_y_te.npy -> label for black and white images
  
color_data.zip contains
- mnist_color.pkl -> colored training images
- mnist_color_te.pkl -> colored test images
- mnist_color_y_te.npy -> label for colored images

## Features
- Trains a VAE on MNIST color or MNIST black-and-white  
- Clean object-oriented design  
- Custom KL divergence and log-likelihood functions  
- Reparameterization trick implementation  
- Latent space visualization using t-SNE  
- Automatic dataset downloading and batching  
- Command-line interface using argparse  

## How to Run
Train the VAE:
python train_vae.py --dset mnist_bw --epochs 10

Show available arguments:
python train_vae.py -h

## Outputs
The script automatically generates:
- reconstructed images  
- latent space plots  
- sample generations from the decoder  

## Requirements
- tensorflow
- numpy
- matplotlib
- sklearn
- pickle
- wget (for downloading files which are also provided here)

