import pytorch_lightning as pl
import torchmetrics
from torchmetrics import F1Score, JaccardIndex, MetricCollection, ConfusionMatrix
from torchmetrics.wrappers import MultitaskWrapper, ClasswiseWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau, OneCycleLR, PolynomialLR
from torchvision import models

class VGG16(pl.LightningModule):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()

        self.model_name = kwargs.get('model_name')
        self.lr = kwargs.get('lr')
        self.num_classes = kwargs.get('num_classes')
        self.lrs_step_size = kwargs.get('lrs_step_size')
        self.lrs_gamma = kwargs.get('lrs_gamma')
        self.class_labels = kwargs.get('class_labels')

        self.save_hyperparameters()
        self.hparams.class_labels = self.class_labels if isinstance(self.class_labels, list) else self.class_labels.replace(" ", "").split(",")

        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.model.classifier[6] = torch.nn.Linear(4096, self.num_classes)

        self.metric = MetricCollection({
            "F1": F1Score('multiclass', num_classes=self.num_classes, average='macro'),
            # "IoU": JaccardIndex('multiclass', num_classes=self.num_classes, average='macro'),
            "class_F1": ClasswiseWrapper(F1Score('multiclass', num_classes=self.num_classes, average=None), labels=self.hparams.class_labels, prefix="F1_"),
            # "class_IoU": ClasswiseWrapper(JaccardIndex('multiclass', num_classes=self.num_classes, average=None), labels=self.hparams.class_labels, prefix="IoU_"),
        })

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)

        loss = nn.CrossEntropyLoss(reduction='none')(logits, y).mean()

        # self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss(reduction='none')(logits, y).mean()
        
        preds = torch.argmax(logits, dim=1)
        self.metric(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log_dict(self.metric.compute())

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss(reduction='mean')(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.metric(preds, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log_dict(self.metric.compute())

        return loss
    
    def configure_optimizers(self):
        # Define and return the optimizer(s) and scheduler(s)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = StepLR(optimizer, step_size=self.hparams.lrs_step_size, gamma=self.hparams.lrs_gamma, verbose=True)
        return [optimizer], [lr_scheduler]
