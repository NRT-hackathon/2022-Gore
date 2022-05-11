import argparse
import torch.nn as nn
import torch.nn.parallel
import pytorch_lightning as lightning

from iolocal import *
import os
import numpy as np
from sklearn.ensemble._hist_gradient_boosting import loss
from scipy.stats import wasserstein_distance


class Discriminator(nn.Module):
    """Discriminator network
    
    Attributes
    ----------
    image_size: int
        Image size. We assume a cubic volume

    nz: int
        Length of the latent space vector

    n_channels: int
        Number of channels in input image

    ndf: int
        Number of discriminator feature maps
    """

    def __init__(self, image_size, n_channels, ndf):
        super(Discriminator, self).__init__()
        
        if image_size % 16 != 0:
            raise ValueError("image_size has to be a multiple of 16")

        # Starting block
        conv_blocks = [ 
            self._make_block(n_channels, ndf, batch_norm=False)
        ]

        csize = image_size / 2
        cndf = ndf
        while csize > 4:
            inc = cndf
            outc = cndf * 2
            conv_blocks.append(
                self._make_block(inc, outc, bias=False)
            )
            cndf *= 2
            csize /= 2

        # Ending block
        conv_blocks.append(
                self._make_block(cndf, 1, kernel_size=4, stride=1, padding=0,
                                 bias=False, last_block=True)
        )

        self.disc = nn.Sequential(*conv_blocks)

    def _make_block(self, in_channels, out_channels,
                    kernel_size=4, stride=2,
                    padding=1, bias=False,
                    batch_norm=True, last_block=False):
        
        if not last_block:
            block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=1),
                nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            #block = nn.Sequential(
            #    nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=1),
            #    nn.Sigmoid()
            #)
            block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=1),
                nn.Linear(out_channels, 1, bias=bias)
            )       # Wasserstein ends with Linear, not Sigmoid, layer
            
        
        return block

    def forward(self, x):
        return self.disc(x).view(-1, 1).squeeze(1)


class Generator(nn.Module):
    """Generator network

    """

    def __init__(self, image_size, n_channels, ngf, nz):
        super(Generator, self).__init__()
        
        if image_size % 16 != 0:
            raise ValueError("image_size has to be a multiple of 16")
        
        cngf, timage_size = ngf // 2, 4
        while timage_size != image_size:
            cngf = cngf * 2
            timage_size = timage_size * 2

        conv_blocks = [
            self._make_block(nz, cngf, 4, 1, 0, bias=False)  # original
        ]
        
        csize = 4
        while csize < image_size // 2:
            conv_blocks.append(
                self._make_block(cngf, cngf // 2)
            )
            cngf = cngf // 2
            csize = csize * 2

        conv_blocks.append(
            self._make_block(cngf, n_channels, 4, 2, 1, bias=False, last_block=True)
        )

        self.gen = nn.Sequential(*conv_blocks)
        
    @staticmethod
    def _make_block(
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        bias: bool=False,
        last_block: bool=False,
    ):
        
        if not last_block:
            block = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(True)
            )

        else:
            block = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=1),
                nn.Tanh()
            )

        return block

    def forward(self, noise):
        return self.gen(noise)


class DCGAN3D(lightning.LightningModule):
    """DCGAN 3D implementation.

    This module is implemented as a pytorch lightning module

    Attributes
    ----------
    generator: torch.nn.Module
        The generator neural network

    discriminator: torch.nn.Module
        The discriminator neural network

    <strike>criterion: torch.nn._Loss
        Loss criterion </strike>

    hparams: 
        Model hyperparameters that are passed to the __init__ function
        and saved in the model state by lightning

    """


    # def __init__(self, ngf=64, ndf=64, image_channels=1, nz=128, learning_rate=2e-4, latent_dim=32, image_size=64, **kwargs):
    def __init__(self, ngf=64, ndf=64, image_channels=1, nz=64, learning_rate=0.00005, latent_dim=32, image_size=64, load_model_pth='n', **kwargs):
        """
        Parameters
        ----------
        <strike> beta1: float
            Beta1 value for Adam optimizer </strike>
        
        ngf: int 
            Number of feature maps to use for the generator
        
        ndf: int
            Number of feature maps to use for the discriminator
        
        image_channels: int
            Number of channels of the images from the dataset
            
        nz: int
            Dimension of the latent space
            
        learning_rate: float
            Learning rate
            
        load_model_pth: str
            Where to find pre-made models if we are continuing training
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()

        self.save_path = './models_done'
        
        self.opt_disc = None
        self.opt_gen = None
        
        self.epoch_counting = 0
        
        self.best_loss = np.inf
        
        if os.getenv("SLURM_JOB_ID") is not None:
            self.slurm_job_id = os.environ["SLURM_JOB_ID"]
            print("--------------------------------")
            print("slurm_job_id: " + self.slurm_job_id)
            print("--------------------------------")
        else:
            print(">>> Not running on an HPC.")
            
        print(self.hparams.keys())
        
        self.load_model_pth = load_model_pth
        
        # Call train.py with "--load_model_path <location and name of saved model>" to start training from a saved model
        # current method idea from: https://discuss.pytorch.org/t/saving-model-and-optimiser-and-scheduler/52030/8
        if self.load_model_pth != 'n':
            print("Loading model from: '" + self.load_model_pth + "'")
            checkpoint = torch.load(self.load_model_pth)
            print(list(checkpoint.keys()))
            self.load_state_dict(checkpoint['model_state_dict']),
            self.opt_disc = checkpoint['disc_optimizer'],
            self.opt_gen = checkpoint['gen_optimizer'],
            self.epoch_counting = int(checkpoint['epoch']) + 1


    def _get_generator(self):
        generator = Generator(self.hparams.image_size, self.hparams.image_channels, self.hparams.ngf, self.hparams.nz)
        generator.apply(self._weights_init)
        
        return generator


    def _get_discriminator(self):
        discriminator = Discriminator(self.hparams.image_size, self.hparams.image_channels, self.hparams.ndf)
        discriminator.apply(self._weights_init)
        
        return discriminator


    @staticmethod
    def _weights_init(m):
        """Initialize the weights of the network"""
        
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    def configure_optimizers(self):
        """Configure the optimizer parameters"""
        lr = self.hparams.learning_rate

        # W-gan uses RMSprop, per the Arjovsky paper        
        self.opt_disc = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)
        self.opt_gen = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        return [self.opt_disc, self.opt_gen], []


    def forward(self, noise):
        """Generates an image given input noise
        
        Parameters
        ----------
        noise: torch.Tensor
            Random noise 
        """
        
        return self.generator(noise)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(real)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(real)

        return result

    
    def validation_step(self, batch, batch_idx):
        print()
        print("Validation ")
        result = self._gen_step(batch)
        return result

    def _disc_step(self, real):
        disc_loss = self._get_disc_loss(real)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def _gen_step(self, real):
        gen_loss = self._get_gen_loss(real)
        self.log("loss/gen", gen_loss, on_epoch=True)
        return gen_loss

    def _get_disc_loss(self, real):
        # Argument:
        #   real    A set of real samples
        #
        # Wasserstein loss for discriminator/critic
        
        ## Generate a set of FAKE outputs, get the discriminator's reaction to them, then do the same with the inputed set of REAL inputs
        fake_pred = self._get_fake_pred(real)
        real_pred = self.discriminator(real)

        # Based on https://agustinus.kristia.de/techblog/2017/02/04/wasserstein-gan/ 
        return -(torch.mean(real_pred) - torch.mean(fake_pred))


    def _get_gen_loss(self, real):
        # Argument:
        #   real    A set of real samples
        #
        # Wasserstein loss for generator
    
        # Generate a set of fake outputs
        fake_pred = self._get_fake_pred(real)
        
        # Based on https://agustinus.kristia.de/techblog/2017/02/04/wasserstein-gan/ 
        return -torch.mean(fake_pred)


    def _get_fake_pred(self, real):
        batch_size = len(real)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake = self.generator(noise)
        fake_pred = self.discriminator(fake)

        return fake_pred

    def _get_noise(self, n_samples, latent_dim):
        batch_size_loc = 16
        
        rand_input = torch.randn(batch_size_loc * 64 * 1 * 1 * 1, device=self.device)
        rand_input = rand_input.view(batch_size_loc, 64, 1, 1, 1)
        
        return rand_input
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--beta1", default=0.5, type=float)
        parser.add_argument("--ngf", default=64, type=int)
        parser.add_argument("--ndf", default=64, type=int)
        parser.add_argument("--nz", default=64, type=int)
        parser.add_argument("--learning_rate", default=0.0002, type=float)
        
        return parser

    def validation_epoch_end(self, outputs):
        
        # Clip parameters in discriminator to be between -0.01 and +0.01, per the Arjovsky WGAN paper.        
        for paramtr in self.discriminator.parameters():
            paramtr.data.clamp_(-0.01, 0.01)
        
        # Begin saving off a generated volume as a numpy array.
        batch_size_loc = 1
        rand_input = torch.randn(batch_size_loc * 64 * 1 * 1 * 1)
        rand_input = rand_input.view(batch_size_loc, 64, 1, 1, 1)
        
        output = self.generator(rand_input)
        output = output.detach().numpy()
        
        file_name = str(self.slurm_job_id) + "_generatedvolume_" + str(self.epoch_counting) + '.npy'
        np.save(os.path.join(self.save_path, file_name), output)
        #print("numpy matrix of generated volume saved to: " + file_name)
        # Done saving off a generated volume

        # Save off the model.
        torch.save({
                'model_state_dict': self.state_dict(),
                'disc_optimizer': self.opt_disc,
                'gen_optimizer': self.opt_gen,
                'loss_info': outputs,
                'epoch': self.epoch_counting
                }, os.path.join(self.save_path, str(self.slurm_job_id) + '_model_and_opt_save_' + str(self.epoch_counting) + '.torchsave'))
        
        self.epoch_counting += 1


def cli_main(args, parser=None, debug=False):

    # Set seed for debugging runs
    if debug:
        lightning.seed_everything(1234)

    if not parser:
        parser = argparse.ArgumentParser()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--num_workers", default=64, type=int)
    parser.add_argument("--latent_dim", default=32, type=int)
    parser.add_argument("--load_model_pth", default='n', type=str)

    # Add model specific arguments
    parser = DCGAN3D.add_model_specific_args(parser)

    # Add lightning.Trainer arguments
    parser = lightning.Trainer.add_argparse_args(parser)

    print(" &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& args: " + str(args))

    # Parse arguments
    args = parser.parse_args(args)
    
    print(" #################################################################### args 2 " + str(args))

    # Setup the dataloader/datamodule
    datamodule = DRPDataModule()

    # Setup[ the model
    if args.load_model_pth != 'n':
        print("Requesting load of pretrained model from " + args.load_model_pth)
        model = DCGAN3D(load_model_pth=args.load_model_pth)
    else:
        print("Instantiating new GAN")
        model = DCGAN3D()

    model.to(device)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print("Running on CPU.")
    
    print(model)

    # Setup the trainer
    trainer = lightning.Trainer().from_argparse_args(args)
    #trainer = lightning.Trainer(gpus=3).from_argparse_args(args)

    # Fit the model
    trainer.fit(model, datamodule)
