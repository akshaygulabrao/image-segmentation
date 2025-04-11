import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from network import MobileNetV2Segmentation, VOC2012DataModule

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

def visualize_multi_class_masks(original_image, true_mask, pred_mask, num_classes=21):
    """
    Visualize the original image, true mask, predicted mask, and correctness for multi-class segmentation
    
    Args:
        original_image: Original input image (1, 3, H, W)
        true_mask: Ground truth mask (1, 1, H, W)
        pred_mask: Predicted mask (1, num_classes, H, W)
        num_classes: Number of classes in the dataset
    """
    # Convert tensors to numpy arrays
    img = original_image.squeeze().permute(1, 2, 0).numpy()
    true_mask = true_mask.squeeze().numpy()
    
    # Get predicted class (argmax across channels)
    pred_class = pred_mask.squeeze().argmax(dim=0).numpy()
    
    # Create masks ignoring void (255) and background (0)
    valid_pixels = (true_mask != 255) & (true_mask != 0)
    
    # Create binary masks for visualization
    true_valid = true_mask.copy()
    true_valid[~valid_pixels] = 0  # Set invalid pixels to background
    
    pred_valid = pred_class.copy()
    pred_valid[~valid_pixels] = 0  # Set invalid pixels to background
    
    # Calculate correct and incorrect pixels
    correct = (true_valid == pred_valid) & valid_pixels
    incorrect = (true_valid != pred_valid) & valid_pixels
    
    # Create visualization of correctness
    correctness_map = np.zeros((*true_mask.shape, 3))
    correctness_map[correct] = [0, 1, 0]  # Green for correct
    correctness_map[incorrect] = [1, 0, 0]  # Red for incorrect
    
    # Get unique classes present in true and predicted masks (excluding 0 and 255)
    true_unique = np.unique(true_valid)
    pred_unique = np.unique(pred_valid)
    true_unique = true_unique[(true_unique != 0) & (true_unique != 255)]
    pred_unique = pred_unique[(pred_unique != 0) & (pred_unique != 255)]
    
    # Convert class indices to human-readable labels
    true_labels = [voc_labels[idx] for idx in true_unique]
    pred_labels = [voc_labels[idx] for idx in pred_unique]
    
    # Create class visualization
    # Create a colormap for the classes (first color for background)
    colors = plt.cm.get_cmap('tab20', num_classes)(np.arange(num_classes))
    colors[0] = [0, 0, 0, 1]  # Set background to black
    class_cmap = ListedColormap(colors)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot original image
    axes[0,0].imshow(img)
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    # Plot true mask with class labels in title
    true_title = 'True Mask\nClasses: ' + ', '.join(true_labels) if true_labels else 'True Mask\nNo valid classes'
    axes[0,1].imshow(true_valid, cmap=class_cmap, vmin=0, vmax=num_classes-1)
    axes[0,1].set_title(true_title)
    axes[0,1].axis('off')
    
    # Plot predicted mask with class labels in title
    pred_title = 'Predicted Mask\nClasses: ' + ', '.join(pred_labels) if pred_labels else 'Predicted Mask\nNo valid classes'
    axes[1,0].imshow(pred_valid, cmap=class_cmap, vmin=0, vmax=num_classes-1)
    axes[1,0].set_title(pred_title)
    axes[1,0].axis('off')
    
    # Plot correctness
    axes[1,1].imshow(img)
    axes[1,1].imshow(correctness_map, alpha=0.5)
    axes[1,1].set_title('Correctness\nGreen=Correct, Red=Incorrect')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_module = VOC2012DataModule(batch_size=4, num_workers=0)
    data_module.setup()

    val_dataset = data_module.val_dataset

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        transforms.Resize((256, 256))
    ])
    
    X_original, y = val_dataset[0]
    X_original = X_original.unsqueeze(0)
    y = y.unsqueeze(0)
    X = transform(X_original)

    print(X.shape, y.shape)
    
    # Load the checkpoint
    model = MobileNetV2Segmentation.load_from_checkpoint(
        checkpoint_path="lightning_logs/version_6/checkpoints/epoch=9-step=360.ckpt"
    )

    # Optional: if you want to use it for inference only
    model.eval()

    with torch.no_grad():
        y_hat = model(X)
    print(y_hat.shape)
    
    # Visualize the results
    visualize_multi_class_masks(X_original, y, y_hat, num_classes=21)