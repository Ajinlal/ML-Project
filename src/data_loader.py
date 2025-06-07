import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image # Explicitly import Image from PIL

class ImageFolderWithPaths(datasets.ImageFolder):
    """
    A custom ImageFolder dataset that returns image paths along with images and labels.
    Useful for debugging and tracking specific misclassifications.
    """
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0] # self.imgs is a list of (image_path, class_id) tuples
        return original_tuple[0], original_tuple[1], path

def create_data_loaders(train_dir, val_dir, test_dir, batch_size, img_size, include_paths=False, num_workers=4):
    # Define image transformations for different phases
    # Common transformations for all models (Resize, ToTensor, Normalize)
    # Data augmentation (RandomHorizontalFlip, RandomRotation) only for training
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(), # Converts PIL image to PyTorch Tensor (H, W, C) to (C, H, W) and normalizes to [0,1]
        # Normalization using ImageNet means and stds - standard practice for pre-trained models
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    common_transforms = transforms.Compose([ # For Validation and Test
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    DatasetClass = ImageFolderWithPaths if include_paths else datasets.ImageFolder

    # Create datasets, checking if directories exist and are not empty
    train_dataset = None
    if train_dir and os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
        # Check if the directory has actual image files or subdirectories with images
        # This is a basic check; ImageFolder will handle actual image loading
        if any(os.path.isdir(os.path.join(train_dir, d)) for d in os.listdir(train_dir)):
            train_dataset = DatasetClass(train_dir, transform=train_transforms)
        else:
            print(f"Warning: Training directory '{train_dir}' exists but seems to contain no class subfolders. Training will be skipped.")
    else:
        print(f"Warning: Training directory '{train_dir}' is not found or empty. Training will be skipped.")

    val_dataset = None
    if val_dir and os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0:
        if any(os.path.isdir(os.path.join(val_dir, d)) for d in os.listdir(val_dir)):
            val_dataset = DatasetClass(val_dir, transform=common_transforms)
        else:
            print(f"Warning: Validation directory '{val_dir}' exists but seems to contain no class subfolders. Validation will be skipped.")
    else:
        print(f"Warning: Validation directory '{val_dir}' is not found or empty. Validation will be skipped.")

    test_dataset = None
    if test_dir and os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
        if any(os.path.isdir(os.path.join(test_dir, d)) for d in os.listdir(test_dir)):
            test_dataset = DatasetClass(test_dir, transform=common_transforms)
        else:
            print(f"Warning: Test directory '{test_dir}' exists but seems to contain no class subfolders. Testing will be skipped.")
    else:
        print(f"Warning: Test directory '{test_dir}' is not found or empty. Testing will be skipped.")


    # Create data loaders
    # num_workers > 0 can speed up data loading, but set to 0 if you hit issues on Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if test_dataset else None

    # Determine class names from any available dataset
    class_names = []
    if train_dataset:
        class_names = train_dataset.classes
    elif val_dataset:
        class_names = val_dataset.classes
    elif test_dataset:
        class_names = test_dataset.classes
    
    return train_loader, val_loader, test_loader, class_names

if __name__ == '__main__':
    # Test data loaders locally
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root is the directory above 'src'
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    
    train_data_dir = os.path.join(project_root, 'data', 'train')
    val_data_dir = os.path.join(project_root, 'data', 'val')
    test_data_dir = os.path.join(project_root, 'data', 'test')

    batch_size = 32
    img_size = (224, 224)

    print(f"Checking data directories:\nTrain: {train_data_dir}\nVal: {val_data_dir}\nTest: {test_data_dir}")

    # Pass include_paths=True to get paths for debugging if needed
    # num_workers=0 for simple testing on Windows to avoid multiprocessing issues
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        train_data_dir, val_data_dir, test_data_dir, batch_size, img_size, include_paths=True, num_workers=0
    )

    print(f"Class names found: {class_names}")
    print("Number of training samples:", len(train_loader.dataset) if train_loader else 0)
    print("Number of validation samples:", len(val_loader.dataset) if val_loader else 0)
    print("Number of test samples:", len(test_loader.dataset) if test_loader else 0)

    if train_loader and len(train_loader.dataset) > 0:
        print("\nFetching one batch from train_loader...")
        images, labels, paths = next(iter(train_loader))
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print(f"First image path in batch: {paths[0]}")
    else:
        print("\nTrain loader not available for testing or is empty.")

    if val_loader and len(val_loader.dataset) > 0:
        print("\nFetching one batch from val_loader...")
        images, labels, paths = next(iter(val_loader))
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
    else:
        print("\nValidation loader not available for testing or is empty.")
    
    if test_loader and len(test_loader.dataset) > 0:
        print("\nFetching one batch from test_loader...")
        images, labels, paths = next(iter(test_loader))
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
    else:
        print("\nTest loader not available for testing or is empty.")
        