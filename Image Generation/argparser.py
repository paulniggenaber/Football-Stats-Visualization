import argparse
## defines the parser an output when 'python train_vae.py -h' is passed in the terminal
#  @return parser.parse_args(): full argument parser
def get_args():
    parser = argparse.ArgumentParser(
                                    prog='train_vae.py',
                                    description=('This code trains a VAE. The arguments will train, visualize, or generate samples from the model.'),
                                    epilog=('Example: python train_vae.py --dset mnist_bw --epochs 50 --visualize_latent --generate_from_prior --generate_from_posterior')
                                    )
    parser.add_argument('--dset', type=str, required=True, choices=['mnist_bw', 'mnist_color'], help='choose dataset from {mnist_bw, mnist_color}')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--visualize_latent', action='store_true', help='visualize latent space')
    parser.add_argument('--generate_from_prior', action='store_true', help='sample images from prior N(0, I)')
    parser.add_argument('--generate_from_posterior', action='store_true', help='sample images from posterior q(z|x)')
    return parser.parse_args()