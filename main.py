import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from network import MobileNetV2Segmentation, VOC2012DataModule

def visualize_masks(original_image, true_mask, pred_mask, class_idx=1):
    """
    Visualize the original image, true mask, predicted mask, and correctness of prediction
    
    Args:
        original_image: Original input image (1, 3, H, W)
        true_mask: Ground truth mask (1, 1, H, W)
        pred_mask: Predicted mask (1, num_classes, H, W)
        class_idx: Class index to visualize (default 1 for foreground)
    """
    # Convert tensors to numpy arrays
    img = original_image.squeeze().permute(1, 2, 0).numpy()
    true_mask = true_mask.squeeze().numpy()
    
    # Get predicted class (argmax across channels)
    pred_class = pred_mask.squeeze().argmax(dim=0).numpy()
    
    # Create binary masks for visualization
    true_binary = (true_mask == class_idx)
    pred_binary = (pred_class == class_idx)
    
    # Calculate correct and incorrect pixels
    correct = (true_binary & pred_binary)  # True positives
    false_pos = (~true_binary & pred_binary)  # False positives
    false_neg = (true_binary & ~pred_binary)  # False negatives
    
    # Create RGB visualization of correctness
    correctness_map = np.zeros((*true_binary.shape, 3))
    correctness_map[correct] = [0, 1, 0]  # Green for correct
    correctness_map[false_pos] = [1, 0, 0]  # Red for false positive
    correctness_map[false_neg] = [0, 0, 1]  # Blue for false negative
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot true mask
    axes[1].imshow(true_binary, cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')
    
    # Plot predicted mask
    axes[2].imshow(pred_binary, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    # Plot correctness
    axes[3].imshow(img)
    axes[3].imshow(correctness_map, alpha=0.5)
    axes[3].set_title('Correctness (Green=Correct, Red=FP, Blue=FN)')
    axes[3].axis('off')
    
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
    visualize_masks(X_original, y, y_hat)