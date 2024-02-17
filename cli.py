import click
import yaml
import importlib
import pytorch_lightning as pl
import torch
import os
import torchvision
from utils.config import Config
from pytorch_lightning import seed_everything
import wandb

seed_everything(42, workers=True)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True, dir_okay=False))
def train(config):
    """
    Train a PyTorch Lightning model using the provided config file.
    """
    config = Config(config)
    
    wandb.login(key=config.get_wandb_key())
    
    # Create DataModule instance
    datamodule = config.get_datamodule()

    # Create Model instance
    model = config.get_modelmodule()

    # Get trainer config
    trainer_config = config.get_trainer_config()

    trainer = pl.Trainer(**trainer_config)
    
    # Train the model
    trainer.fit(model, datamodule, ckpt_path=config.get_last_checkpoint())

# Define the test command (unchanged)
@cli.command()
@click.option('--config', type=click.Path(exists=True, dir_okay=False))
def test(config):
    """
    Test a PyTorch Lightning model using the provided config file.
    """
    config = Config(config)
    
    # Create DataModule instance
    datamodule = config.get_datamodule()

    # Create Model instance
    model = config.get_modelmodule()

    # Get trainer config
    trainer_config = config.get_test_trainer_config()

    trainer = pl.Trainer(**trainer_config)
    
    # Test the model
    trainer.test(model, datamodule, ckpt_path=config.get_best_checkpoint())

if __name__ == '__main__':
    cli()