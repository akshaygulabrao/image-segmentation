import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import VOCSegmentation
from torchmetrics import Accuracy, JaccardIndex

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner

class VOC2012DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size= batch_size
        self.num_workers = os.cpu_count() // 2 - 1
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
            transforms.Resize((256, 256))
        ])
        
    def setup(self, stage=None):
        self.train_dataset = VOCSegmentation(
            root='./data',
            year='2012',
            image_set='train',
            download=True,
            transform=self.transform,
            target_transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.PILToTensor()
            ])
        )
        
        self.val_dataset = VOCSegmentation(
            root='./data',
            year='2012',
            image_set='val',
            download=True,
            transform=self.transform,
            target_transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.PILToTensor()
            ])
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def main():
    dry_run = os.getenv('DRY_RUN', '0') == '1'
    gpus = os.getenv('GPUS','0') == '1'
    wandb_logger = WandbLogger(project="image-segmentation",log_model=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",          # Track validation IoU
        mode="max",                 # Save model with max IoU
        dirpath="checkpoints",      # Local dir (optional)
        filename="best_model_{epoch}_{val_iou:.2f}",
        save_top_k=1,               # Save only the best model
        save_last=True
    )

    trainer_config = {
        'max_epochs': 10 if dry_run else 50,
        'accelerator': 'auto',
        'devices': 'auto',
        'enable_progress_bar': True,
        'log_every_n_steps': 10,
        'overfit_batches': 5 if dry_run else 0,  # Overfit 5 batches in dry run mode
        'logger': wandb_logger,
        'callbacks' : [checkpoint_callback]
    }
    
    trainer = pl.Trainer(**trainer_config)

    model = MobileNetV2Segmentation(num_classes=21, learning_rate=1e-3)
    data_module = VOC2012DataModule(batch_size=32)
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()