import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from network import MobileNetV2Segmentation,VOC2012DataModule


def visualize_sample(images, masks, model):
    """
    Visualizes samples from the dataloader with class labels.
    
    Args:
        images: Batch of images (N, C, H, W)
        masks: Batch of masks (N, H, W)
        num_samples: Number of samples to display
    """
    # VOC class labels mapping
    voc_labels = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'potted plant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tv/monitor',
        255: 'unlabeled'
    }
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    
    # Convert to numpy and permute dimensions for matplotlib
    images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    masks = masks.detach().cpu().numpy()
    
    fig, axes = plt.subplots(min(1, len(images)), 2, figsize=(10, 10))
    
    for i in range(min(1, len(images))):
        if 1 > 1:
            ax_img = axes[i, 0]
            ax_mask = axes[i, 1]
        else:
            ax_img = axes[0]
            ax_mask = axes[1]
            
        ax_img.imshow(images[i])
        ax_img.set_title('Image')
        ax_img.axis('off')
        
        # Get unique classes in the mask (excluding background)
        unique_classes = np.unique(masks[i])
        present_classes = [c for c in unique_classes if c != 0]
        
        # Create title with present classes
        if len(present_classes) > 0:
            class_names = [voc_labels[c] for c in present_classes]
            mask_title = f'Mask (Classes: {", ".join(class_names)})'
        else:
            mask_title = 'Mask (Background only)'
        
        ax_mask.imshow(masks[i], cmap='jet')
        ax_mask.set_title(mask_title)
        ax_mask.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    data_module = VOC2012DataModule(batch_size=4, num_workers=0)
    data_module.setup()
    
    val_dataset = data_module.val_dataset
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
    )
    X,y = val_dataloader[0]

    print(X.shape, y.shape)
    # Load the checkpoint
    model = MobileNetV2Segmentation.load_from_checkpoint(
        checkpoint_path="lightning_logs/version_6/checkpoints/epoch=9-step=360.ckpt"
    )

    # Optional: if you want to use it for inference only
    model.eval()


    