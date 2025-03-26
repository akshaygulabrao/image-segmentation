import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def get_voc_dataloaders(batch_size=4, img_size=256, data_dir='./data'):
    """
    Returns train and validation dataloaders for Pascal VOC segmentation dataset.
    
    Args:
        batch_size: Number of samples per batch
        img_size: Size to resize images to (square)
        data_dir: Directory to store/download dataset
    Returns:
        train_loader: Training set dataloader
        val_loader: Validation set dataloader
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.int64))
    ])
    
    # Download and load datasets
    train_set = torchvision.datasets.VOCSegmentation(
        root=data_dir,
        year='2012',
        image_set='train',
        download=True,
        transform=transform,
        target_transform=target_transform
    )
    
    val_set = torchvision.datasets.VOCSegmentation(
        root=data_dir,
        year='2012',
        image_set='val',
        download=True,
        transform=transform,
        target_transform=target_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def visualize_sample(images, masks, num_samples=4):
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
    
    fig, axes = plt.subplots(min(num_samples, len(images)), 2, figsize=(10, 10))
    
    for i in range(min(num_samples, len(images))):
        if num_samples > 1:
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
    # Get dataloaders
    train_loader, val_loader = get_voc_dataloaders(batch_size=4, img_size=256)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Visualize a sample batch
    images, masks = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
    
    visualize_sample(images, masks)