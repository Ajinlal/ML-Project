import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import matplotlib.pyplot as plt
import sys

# Add project root to sys.path to resolve imports like 'src.data_loader'
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
sys.path.insert(0, project_root)

# Corrected import path for eyecare_model.py
from src.data_loader import create_data_loaders
from src.eyecare_model import TransferLearningModel # Corrected from models.eyecare_model

def train_model(train_loader, val_loader, class_names, num_epochs, model_name, learning_rate, device):
    num_classes = len(class_names)
    # Pass pretrained=True to load ImageNet pre-trained weights
    model = TransferLearningModel(num_classes=num_classes, model_name=model_name, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Learning rate scheduler: decays learning rate by a factor of gamma every step_size epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) 

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}') # Start epoch count from 1
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            if dataloader is None or len(dataloader.dataset) == 0:
                print(f"Skipping {phase} phase: Dataloader is None or empty. Please check your data directory and splits.")
                continue

            # Iterate over data (expecting inputs, labels, and paths because include_paths=True is set in main)
            for inputs, labels, _ in dataloader: # Paths are not needed for training, so _
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # Get the predicted class
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Step the scheduler after the training phase (once per epoch)
            if phase == 'train':
                scheduler.step()

            # Calculate epoch loss and accuracy
            # Ensure len(dataloader.dataset) is not zero to avoid division by zero
            if len(dataloader.dataset) > 0:
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
            else:
                epoch_loss = 0.0
                epoch_acc = 0.0

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation accuracy
            if phase == 'val':
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save the best model
                    model_save_path = os.path.join(project_root, 'models', 'best_model.pth')
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Best model saved to {model_save_path} with validation accuracy: {best_acc:.4f}")
            elif phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())

    print() # Newline for better readability after each epoch loop

    print(f'Training complete. Best validation Accuracy: {best_acc:.4f}')

    # Plotting training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjusts plot to prevent labels from overlapping
    plt.show()

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # Configuration
    train_data_dir = os.path.join(project_root, 'data', 'train')
    val_data_dir = os.path.join(project_root, 'data', 'val')
    test_data_dir = os.path.join(project_root, 'data', 'test') # Included for create_data_loaders to get class_names

    batch_size = 32
    img_size = (224, 224) # Standard input size for ResNet/AlexNet
    num_epochs = 10 # You can adjust this: more epochs if model is still improving, fewer if overfitting
    learning_rate = 0.001
    model_name = "resnet18" # Or "alexnet"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Create data loaders (include_paths=True is crucial for avoiding ValueError)
    # Set num_workers=0 if you face issues on Windows with multiprocessing
    train_loader, val_loader, _, class_names = create_data_loaders(
        train_data_dir, val_data_dir, test_data_dir, batch_size, img_size, include_paths=True, num_workers=0
    )

    if train_loader is None or val_loader is None or len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        print("ERROR: Training or Validation data loaders could not be created properly or are empty. "
              "Please ensure 'data/train' and 'data/val' exist, are not empty, and contain class subfolders with images.")
    elif not class_names:
        print("ERROR: No class names found. Ensure your data folders contain class subfolders.")
    else:
        print(f"Found {len(class_names)} classes: {class_names}")
        print(f"Training data: {len(train_loader.dataset)} samples")
        print(f"Validation data: {len(val_loader.dataset)} samples")

        # Train the model
        trained_model = train_model(
            train_loader, val_loader, class_names, num_epochs, model_name, learning_rate, device
        )