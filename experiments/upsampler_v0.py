import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import VOCSegmentation
from torchmetrics import Accuracy, JaccardIndex

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner

description = """
dataset: Pascal VOC 2012
network: MobileNet V2 (pretrained on imagenet) with basic upsampling;\
backbone has frozen weights
preprocessing: [imagenet normalization, resize to 256^2]
data augmentation: none
loss fn: cross entropy loss
optimizer: Adam defaults
epochs: 50
"""

class MobileNetV2Segmentation(pl.LightningModule):
    def __init__(self, num_classes=21, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = models.mobilenet_v2(weights='DEFAULT').features
        
        for param in self.backbone[:14].parameters():
            param.requires_grad = False
        
        self.decoder = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, num_classes, kernel_size=3, padding=1),
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        )
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=255)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=255)
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255)

    def forward(self, x):
        features = self.backbone(x)
        out = self.decoder(features)
        return out
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.squeeze(1).long()
        
        logits = self(images)
        loss = F.cross_entropy(logits, masks, ignore_index=255)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, masks)
        iou = self.train_iou(preds, masks)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.squeeze(1).long()
        
        logits = self(images)
        loss = F.cross_entropy(logits, masks, ignore_index=255)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, masks)
        iou = self.val_iou(preds, masks)
        
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        self.log('val_iou', iou, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)