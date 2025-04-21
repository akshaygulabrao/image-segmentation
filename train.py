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
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
import wandb

from experiments.upsampler_v1 import name,description,MobileNetV2Segmentation


class VOC2012DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size= batch_size
        self.num_workers = os.cpu_count() // 2 - 1
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize first to save computation on larger operations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
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
            batch_size=512,
            shuffle=False,
            num_workers=self.num_workers
        )

def main():
    for i in range(5):
        if torch.cuda.is_available():
            # Use Weights & Biases if GPU is available
            wandb_api_key = os.getenv("WANDB_API_KEY")
            os.system(f"wandb login --relogin {wandb_api_key}")
            logger = WandbLogger(project="image-segmentation",group=name,name=f"{name}_run{i}",
                                notes=description, log_model=True)
        else:
            # Fallback to TensorBoard for CPU runs
            logger = TensorBoardLogger("tb_logs", name="cpu_runs")  # Logs to ./tb_logs/cpu_
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_iou",          # Track validation IoU
            mode="max",                 # Save model with max IoU
            dirpath="checkpoints",      # Local dir (optional)
            filename="model_{epoch}_{val_iou:.2f}",
            save_top_k=1
        )
        early_stop_callback = EarlyStopping(
            monitor="val_iou",          # Metric to monitor
            patience=2,                # Number of epochs to wait before stopping
            mode="max",                # We want to maximize the IoU
            min_delta=0.001,           # Minimum change to qualify as improvement
            verbose=True
        )

        trainer_config = {
            'max_epochs': 50,
            'accelerator': 'auto',
            'devices': 'auto',
            'enable_progress_bar': True,
            'log_every_n_steps': 10,
            'logger': logger,
            'callbacks' : [checkpoint_callback, early_stop_callback]
        }
        
        trainer = pl.Trainer(**trainer_config)

        model = MobileNetV2Segmentation(num_classes=21)
        data_module = VOC2012DataModule(batch_size=32)
        trainer.fit(model, datamodule=data_module)
        wandb.finish()

if __name__ == '__main__':
    main()