from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.callbacks.spectrogram_callback import LogSpectrogramCallback
from src.models.spectrogram_vae import SpectrogramVAE

DAFX_TO_USE = [
    'mda Delay',
    'mda Overdrive',
    'mda Ambience',
    'mda RingMod',
    'mda Combo',
    'mda Dynamics',
]

SEED = 123
MAX_EPOCHS = 800

if __name__ == "__main__":
    wandb.require("service")
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # arg parse for config
    parser = ArgumentParser()

    # Add available trainer args and system args
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SpectrogramVAE.add_model_specific_args(parser)

    # Parse
    args = parser.parse_args()

    # callbacks
    wandb_logger = WandbLogger(name='vtck_5fx_plsnl_short_wind', project='l5proj_spectrogram_vae')
    # wandb_logger = None

    val_checkpoint = ModelCheckpoint(
        monitor="val_loss/loss",
        filename="{epoch}-{step}",
        mode="min"
    )
    recon_checkpoint = ModelCheckpoint(
        monitor="val_loss/reconstruction_loss",
        filename="best_recon-{epoch}-{step}",
        mode="min"
    )
    kl_checkpoint = ModelCheckpoint(
        monitor="val_loss/kl_divergence",
        filename="best_kldiv-{epoch}-{step}",
        mode="min"
    )

    # Change settings for training
    args.dafx_names = DAFX_TO_USE
    args.beta_end_epoch = MAX_EPOCHS

    # Set up trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        callbacks=[
            LogSpectrogramCallback(),
            val_checkpoint,
            recon_checkpoint,
            kl_checkpoint,
            # early_stopping
        ],
        num_sanity_val_steps=0,
        max_epochs=MAX_EPOCHS,
        accelerator='gpu',
        gradient_clip_val=5.
    )

    # create the System
    system = SpectrogramVAE(**vars(args))

    print(system)

    # train!
    trainer.fit(system)
