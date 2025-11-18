import tensorflow as tf
from tensorflow.keras import layers
from losses import kl_divergence, log_diag_mvn
from tensorflow.keras.models import Sequential
import numpy as np

## BiCoder class inheriting from layers.Layer
class BiCoder(layers.Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x):
        raise NotImplementedError
    
    def process_output(self):
        raise NotImplementedError
    
    ## samples a value using the location scale approach
    # @param mu: mean of learned distribution
    # @param std: standard deviation of learned distribution
    # @return sample: sampled value
    def sample(self, mu, std):
        eps = tf.random.normal(shape=tf.shape(mu))
        sample = mu + std * eps
        return sample

## Encoder class inheriting from BiCoder and to BWEncoder and ColorEncoder
class Encoder(BiCoder):
    def __init__(self):
        super().__init__()
    
    def call(self, x):
        raise NotImplementedError
    
    ## processes the output of the encoder neural networks
    #  @param out: output of encoder neural network
    #  @return z: sampled values
    #  @return mu: mean of learned distribution
    #  @return std: standard deviation of learned distribution
    def process_output(self, out):
        mu  = out[:,:self._latent_dim]
        log_var = out[:,self._latent_dim:]
        std = tf.math.exp(0.5*log_var)
        z = super().sample(mu, std)
        return z, mu, log_var

## Decoder class inheriting from BiCoder and to BWDecoder and ColorDecoder  
#  initializes fixed standard deviation of 0.75 as class variable
class Decoder(BiCoder):
    _std = 0.75
    def __init__(self):
        super().__init__()
    
    def call(self, x):
        raise NotImplementedError

    ## processes the output of the decoder neural networks
    #  @param out: output of decoder neural network
    #  @return x_hat: sampled values
    #  @return mu: mean of learned distribution
    #  @return std: standard deviation of learned distribution
    def process_output(self, out):
        mu = out
        log_var = tf.math.log(self._std**2)
        log_var = tf.cast(log_var, tf.float32)
        x_hat = super().sample(mu, self._std)
        return x_hat, mu, log_var

## ColorEncoder class with latent dimension and neural network
class ColorEncoder(Encoder):
    def __init__(self):
        super().__init__() 
        self._latent_dim = 50
        self._encoder_conv = Sequential(
                                [
                                layers.InputLayer(input_shape=(28,28,3)),
                                layers.Conv2D(
                                filters=32,   kernel_size=3, strides=2, activation='relu', padding='same'),
                                layers.Conv2D(
                                filters=2*32, kernel_size=3, strides=2, activation='relu', padding='same'),
                                layers.Conv2D(
                                filters=4*32, kernel_size=3, strides=2, activation='relu', padding='same'),
                                layers.Flatten(),
                                layers.Dense(2*self._latent_dim)
                                ]
                                )

    ## calls the color encoder and provides the output 
    #  @param x: a batch of data
    #  @return z: sampled values
    #  @return mu: mean of learned distribution
    #  @return std: standard deviation of learned distribution
    def call(self, x):
        out = self._encoder_conv(x)
        z, mu, log_var = super().process_output(out)
        return z, mu, log_var

## ColorDecoder class with latent dimension and neural network
class ColorDecoder(Decoder):
    def __init__(self):
        super().__init__()
        self._latent_dim = 50
        self._decoder_conv = Sequential(
                                [
                                layers.InputLayer(input_shape=self._latent_dim),
                                layers.Dense(units=np.prod((4,4,128)), activation='relu'),
                                layers.Reshape(target_shape=(4,4,128)),
                                layers.Conv2DTranspose(
                                    filters=32*2, kernel_size=3, strides=2, padding='same',output_padding=0,
                                    activation='relu'),
                                layers.Conv2DTranspose(
                                    filters=32, kernel_size=3, strides=2, padding='same',output_padding=1,
                                    activation='relu'),
                                layers.Conv2DTranspose(
                                    filters=3, kernel_size=3, strides=2, padding='same', output_padding=1),
                                layers.Activation('linear', dtype='float32'),
                                ]
                                )
    
    ## calls the color decoder and provides output 
    #  @param x: a batch of data
    #  @return x_hat: sampled values
    #  @return mu: mean of learned distribution
    #  @return std: standard deviation of learned distribution
    def call(self, z):
        out = self._decoder_conv(z)
        x_hat, mu, log_var = super().process_output(out)
        return x_hat, mu, log_var

## BWEncoder class with latent dimension and neural network
class BWEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self._latent_dim = 20
        self._encoder_mlp = Sequential(
                                [ 
                                layers.InputLayer(input_shape=(28*28,)),
                                layers.Dense(units=400,activation='relu'),
                                layers.Dense(2*self._latent_dim),
                                ]
                                )

    ## calls the color encoder and provides the output 
    #  @param x: a batch of data
    #  @return z: sampled values
    #  @return mu: mean of learned distribution
    #  @return std: standard deviation of learned distribution
    def call(self, x):
        out = self._encoder_mlp(x) #encoder for bw
        z, mu, log_var = super().process_output(out)
        return z, mu, log_var

## BWDecoder class with latent dimension and neural network
class BWDecoder(Decoder):
    def __init__(self):
        super().__init__()
        self._latent_dim = 20
        self._decoder_mlp = Sequential(
                                [
                                layers.InputLayer(input_shape=self._latent_dim),
                                layers.Dense(units=400,activation='relu'),
                                layers.Dense(28*28),
                                ]
                                )

    ## calls the color decoder and provides output 
    #  @param x: a batch of data
    #  @return x_hat: sampled values
    #  @return mu: mean of learned distribution
    #  @return std: standard deviation of learned distribution
    def call(self, z):
        out = self._decoder_mlp(z)
        x_hat, mu, log_var = super().process_output(out)
        return x_hat, mu, log_var

## VAE class with encoder, decoder, coders dict, shape variables and data type
class VAE(tf.keras.Model):
    _coders = {'bw': [BWEncoder, BWDecoder, (1, 20)], 'color': [ColorEncoder, ColorDecoder, (3, 50)]}
    def __init__(self, type):
        super().__init__(type)
        encoder_class, decoder_class = VAE._coders[type][0:2]
        self._encoder = encoder_class()
        self._decoder = decoder_class()
        self._c, self._latent_dim = VAE._coders[type][2]
        self._type = type

    ## Calls encoder and decoder, computes the lower bound of the loss
    #  @param x: a batch of data
    #  @return L: loss lower bound
    def call(self, x):
        x = tf.cast(x, tf.float32)
        z, mu_q, log_var_q = self._encoder(x)
        x_hat, mu_p, log_var_p = self._decoder(z)
        L = log_diag_mvn(x, mu_p, log_var_p) - kl_divergence(mu_q, log_var_q) #why log_var and not std
        return L

    ## trains the model once using a batch of data
    #  @param x: a batch of data
    #  @param optimizer: type of optimizer
    @tf.function
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.call(x)
            self._vae_loss = -loss
        gradients = tape.gradient(self._vae_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    ## sample from prior using decoder
    #  @return img: 100 image samples
    def sample_prior(self):
        z = tf.random.normal((100, self._latent_dim))
        x_hat, mu_p, log_var = self._decoder(z)
        img = tf.clip_by_value(tf.reshape(mu_p, (100, 28, 28, self._c)), 0.0, 1.0)
        return img

    ## sample from posterior using decoder
    #  @return img: 100 image samples
    def sample_posterior(self, x):
        z, mu_q, log_var_q = self._encoder(x)
        x_hat, mu_p, log_var_p = self._decoder(z)
        img = tf.clip_by_value(tf.reshape(mu_p, (100, 28, 28, self._c)), 0.0, 1.0)
        return img