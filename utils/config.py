import yaml
import importlib
import pytorch_lightning as pl
import os

class Config:
    def __init__(self, config_filepath):
        with open(config_filepath, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_datamodule(self):
        # Dynamically import the DataModule class from the 'data' directory
        datamodule_module = importlib.import_module('data.' + self.config['datamodule_config']['name'])
        DataModuleClass = getattr(datamodule_module, self.config['datamodule_config']['name'])

        # Create DataModule instance
        return DataModuleClass(**self.config['datamodule_config']['param'])
    
    def get_modelmodule(self):
        # Dynamically import the Model class from the 'models' directory
        model_module = importlib.import_module('models.' + self.config['model_config']['name'])
        ModelClass = getattr(model_module, self.config['model_config']['name'])
        
        # Create Model instance
        return ModelClass(**self.config['model_config']['param'])

    def get_trainer_config(self):
        # Extract callbacks and loggers configurations
        callbacks_config = self.config['trainer_config']['callbacks']
        loggers_config = self.config['trainer_config']['logger']

        # Instantiate callbacks and loggers using the provided configurations
        callbacks = [getattr(pl.callbacks, cb['name'])(**{**cb['param'], "dirpath": self.get_checkpoint_dir()}) for cb in callbacks_config]
        loggers = [getattr(pl.loggers, lg['name'])(**lg['param']) for lg in loggers_config]

        # Create Trainer with custom callbacks and loggers
        trainer_config = self.config['trainer_config']
        trainer_config['callbacks'] = callbacks
        trainer_config['logger'] = loggers

        return trainer_config

    def get_test_trainer_config(self):
        # Extract loggers configurations
        loggers_config = self.config['test_trainer_config']['logger']

        # Instantiate loggers using the provided configurations
        loggers = [getattr(pl.loggers, lg['name'])(**lg['param']) for lg in loggers_config]

        # Create Trainer with custom loggers
        trainer_config = self.config['test_trainer_config']
        trainer_config['logger'] = loggers

        return trainer_config

    def get_checkpoint_dir(self):
        path = os.path.join(
            self.config['project_config']['log_dir'],
            self.config['project_config']['name'],
            self.config['project_config']['version']
        )

        os.makedirs(path, exist_ok=True)

        return path

    def get_best_checkpoint(self):
        path = os.path.join(
            self.get_checkpoint_dir(),
            self.config['project_config']['best_ckpt_name']+'.ckpt',
        )

        if path and os.path.exists(path):
            return path
        return None

    def get_last_checkpoint(self):
        path = os.path.join(
            self.get_checkpoint_dir(),
            'last.ckpt',
        )

        if path and os.path.exists(path):
            return path
        return None

    def get_wandb_key(self):
        return self.config['wandb_key']
