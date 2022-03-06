import argparse
import torch.nn as nn
import torch.nn.parallel
import pytorch_lightning as lightning

from ..io import DRPDataModule


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
        
        if image_size % 16 == 0:
            raise ValueError("image_size has to be a multiple of 16")

        # Starting block
        conv_blocks = [ 
            self._make_block(n_channels, ndf, batch_norm=False)
        ]

        csize = image_size/2
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


    def _make_block(in_channels, out_channels,
                    kernel_size = 4, stride = 2,
                    padding = 1, bias = False,
                    batch_norm = True, last_block = False):
        if not last_block:
            block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm3d(out_channels) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Sigmoid()
            )
        
        return block

    def forward(self, x):
        return self.disc(x).view(-1, 1).squeeze(1)


class Generator(nn.Module):
    """Generator network

    """
    def __init__(self, image_size, n_channels, ngf, nz):
        super(Generator, self).__init__()
        
        if image_size % 16 == 0:
            raise ValueError("image_size has to be a multiple of 16")
        
        cngf, timage_size = ngf//2, 4
        while timage_size != image_size:
            cngf = cngf * 2
            timage_size = timage_size * 2


        conv_blocks = [
            self._make_block(nz, cngf, 4, 1, 0, bias=False)
        ]

        csize = 4
        while csize < image_size//2:
            conv_blocks.append(
                self._make_block(cngf, cngf//2)
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
        kernel_size = 4,
        stride = 2,
        padding = 1,
        bias: bool = False,
        last_block: bool = False,
    ):
        if not last_block:
            block = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        else:
            block = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
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

    criterion: torch.nn._Loss
        Loss criterion

    hparams: 
        Model hyperparameters that are passed to the __init__ function
        and saved in the model state by lightning

    """
    def __init__(self, ngf=64, ndf=64, image_channels=1, nz=100, learning_rate=2e-4, **kwargs):
        """
        Parameters
        ----------
        beta1: float
            Beta1 value for Adam optimizer
        
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
        """
        super().__init__()
        self.save_hyperparameters()

        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()

        self.criterion = nn.BCELoss()


    def _get_generator(self):
        generator = Generator(self.hparams.latent_dim, self.hparams.feature_maps_gen, self.hparams.image_channels)
        generator.apply(self._weights_init)
        return generator


    def _get_discriminator(self):
        discriminator = Discriminator(self.hparams.feature_maps_disc, self.hparams.image_channels)
        discriminator.apply(self._weights_init)
        return discriminator


    @staticmethod
    def _weights_init(m):
        """Initialize the weights of the network"""
        pass


    def configure_optimizers(self):
        """Configure the Adam optimizer parameters"""
        lr = self.hparams.learning_rate
        betas = (self.hparams.beta1, 0.999)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        return [opt_disc, opt_gen], []


    def forward(self, noise):
        """Generates an image given input noise
        
        Parameters
        ----------
        noise: torch.Tensor
            Random noise 
        """
        noise = noise.view(*noise.shape, 1, 1)
        return self.generator(noise)


    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(real)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(real)

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
        # Train with real
        real_pred = self.discriminator(real)
        real_gt = torch.ones_like(real_pred)
        real_loss = self.criterion(real_pred, real_gt)

        # Train with fake
        fake_pred = self._get_fake_pred(real)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        return disc_loss


    def _get_gen_loss(self, real):
        # Train with fake
        fake_pred = self._get_fake_pred(real)
        fake_gt = torch.ones_like(fake_pred)
        gen_loss = self.criterion(fake_pred, fake_gt)

        return gen_loss


    def _get_fake_pred(self, real):
        batch_size = len(real)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake = self(noise)
        fake_pred = self.discriminator(fake)

        return fake_pred


    def _get_noise(self, n_samples, latent_dim):
        return torch.randn(n_samples, latent_dim, device=self.device)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--beta1", default=0.5, type=float)
        parser.add_argument("--ngf", default=64, type=int)
        parser.add_argument("--ndf", default=64, type=int)
        parser.add_argument("--nz", default=100, type=int)
        parser.add_argument("--learning_rate", default=0.0002, type=float)
        return parser


def cli_main(args, parser=None, debug=False):

    # Set seed for debugging runs
    if debug:
        lightning.seed_everything(1234)

    if not parser:
        parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    
    # Add model specific arguments
    parser = DCGAN3D.add_model_specific_args(parser)

    # Add lightning.Trainer arguments
    parser = lightning.Trainer.add_argparse_args(parser)

    # Parse arguments
    args = parser.parse_args(args)

    # Setup the dataloader/datamodule
    # datamodule = DRPDataModule()
    
    # Setup[ the model
    # model = models.DCGAN3D()

    # Setup the trainer
    # trainer = lightning.Trainer.from_argparse_args(args)

    # Fit the model
    # trainer.fit(model, datamodule)
