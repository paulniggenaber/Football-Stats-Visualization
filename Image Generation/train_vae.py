import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE

from dataloader import DataLoader
from vae import VAE
from utils import save_picture
from argparser import get_args

#'python train_vae.py -h' to see argparse description
args = get_args()

loader = DataLoader(args.dset, 'm1')
tr_data = loader.training_data
te_data = loader.test_data
label = loader.label_data
data_type = loader.type
model = VAE(data_type)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
for epoch in range(args.epochs):
    for i, tr_batch in enumerate(tr_data):
        model.train(tr_batch, optimizer)
        if i == 0:
            print(f'Epoch {epoch+1}/{args.epochs}')

## samples from prior 
#  calls a function to save the picture which calls the plotting function
#  @param model: instance of VAE()
#  @param data_type: type of data - 'color' or 'bw'
#  @param epochs: number of epochs used for training 
def generate_from_prior(model, data_type, epochs):
    sample = model.sample_prior()
    save_picture(sample, data_type, epochs, 'prior') #in 'utils.py'

## samples from posterior by getting a batch of the passed data (usually test data)
#  calls a function to save the picture which calls the plotting function 
#  @param model: instance of VAE()
#  @param data: passed data, usually test_data
#  @param data_type: type of data - 'color' or 'bw'
#  @param epochs: number of epochs used for training 
def generate_from_posterior(model, data, data_type, epochs):
    x = next(iter(data))
    sample = model.sample_posterior(x)
    save_picture(sample, data_type, epochs, 'posterior') #in 'utils.py'

## plots a visulaization of the latent space in two dimensions
#  @param model: instance of VAE()
#  @param data: passed data, usually test_data
#  @param label: label data
#  @param data_type: type of data - 'color' or 'bw'
#  @param epochs: number of epochs used for training 
def visualize_latent(model, data, label, data_type, epochs):
    xs = []
    ys = []
    for x_batch, y_batch in zip(data, label):
        xs.append(x_batch.numpy())
        ys.append(y_batch.numpy())

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    x = tf.cast(x, tf.float32)
    # y is type int
    z, mu_q, log_var_q = model._encoder(x)

    tsne = TSNE(n_components=2)
    latent = tsne.fit_transform(z.numpy())

    data_type = 'color' if data_type == 'color' else 'bw'

    plt.figure(figsize=(8, 6))
    plt.scatter(latent[:, 0], latent[:, 1], c=y)
    plt.title(f"Latent Space ({'color' if data_type == 'color' else 'bw'}, {epochs} epochs)")
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig(f'./xhat_latent_{data_type}_{epochs}epochs.pdf')
    plt.close()
    print(f'Saved: ./xhat_latent_{data_type}_{epochs}epochs.pdf')

if args.generate_from_prior:
    generate_from_prior(model, data_type, args.epochs)

if args.generate_from_posterior:
    generate_from_posterior(model, te_data, data_type, args.epochs)

if args.visualize_latent:
    visualize_latent(model, te_data, label, data_type, args.epochs)