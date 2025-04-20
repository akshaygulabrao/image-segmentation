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
data augmentation: lots of geometric and color transformations
loss fn: cross entropy loss
optimizer: Adam defaults
epochs: 50
"""

class VOC2012DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size= batch_size
        self.num_workers = os.cpu_count() // 2 - 1
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize first to save computation on larger operations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
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
    def __init__(self, num_classes=21):
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = models.mobilenet_v2(weights='DEFAULT').features
        
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

def main():

    if torch.cuda.is_available():
        # Use Weights & Biases if GPU is available
        wandb_api_key = os.getenv("WANDB_API_KEY")
        os.system(f"wandb login --relogin {wandb_api_key}")
        logger = WandbLogger(project="image-segmentation",
                             notes=description, log_model=True)
    else:
        # Fallback to TensorBoard for CPU runs
        logger = TensorBoardLogger("tb_logs", name="cpu_runs")  # Logs to ./tb_logs/cpu_
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",          # Track validation IoU
        mode="max",                 # Save model with max IoU
        dirpath="checkpoints",      # Local dir (optional)
        filename="model_{epoch}_{val_iou:.2f}",
        save_last=True
    )

    trainer_config = {
        'max_epochs': 50,
        'accelerator': 'auto',
        'devices': 'auto',
        'enable_progress_bar': True,
        'log_every_n_steps': 10,
        'logger': logger,
        'callbacks' : [checkpoint_callback]
    }
    
    trainer = pl.Trainer(**trainer_config)

    model = MobileNetV2Segmentation(num_classes=21)
    data_module = VOC2012DataModule(batch_size=32)
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()