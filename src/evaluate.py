import sys
import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path for module imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
sys.path.insert(0, project_root)

# Corrected import path for eyecare_model.py
from src.eyecare_model import TransferLearningModel # Corrected from models.eyecare_model
from src.data_loader import create_data_loaders


def evaluate_model(model_path, test_loader, class_names, device):
    if not class_names:
        print("Error: No class names provided for evaluation. Cannot proceed.")
        return

    num_classes = len(class_names)
    # Create model instance with the correct number of classes
    model = TransferLearningModel(num_classes=num_classes).to(device)
    
    try:
        # Load the saved model state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from: {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please ensure the path and filename are correct. "
              "Did you run train.py successfully to save the best_model.pth?")
        return
    except Exception as e:
        print(f"ERROR loading model state dict: {e}. Ensure the model architecture in eyecare_model.py matches the saved model.")
        return

    model.eval() # Set model to evaluation mode (disables dropout, batchnorm updates)

    all_preds = []
    all_labels = []
    
    # Initialize lists for misclassified specific classes, if needed, adjust class names.
    # Assuming 'Glaucoma', 'Myopia', 'Healthy' as examples.
    # Adjust these names based on your actual class_names list.
    misclassified_glaucoma_as_myopia_paths = []
    misclassified_glaucoma_as_healthy_paths = []
    # Add other specific misclassification types as needed

    if test_loader is None or len(test_loader.dataset) == 0:
        print("ERROR: Test/Validation DataLoader could not be created or is empty. "
              "Check data path, folder structure, and ensure the test/validation directory is not empty.")
        return

    print(f"Starting evaluation with {len(test_loader.dataset)} samples across {len(test_loader)} batches...")

    with torch.no_grad(): # Disable gradient calculation for inference
        for i, (inputs, labels, paths) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) # Get the class with the highest probability
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Identify specific misclassifications
            for j in range(len(labels)):
                true_label_idx = labels[j].item()
                predicted_label_idx = preds[j].item()
                image_path = paths[j] # Path from ImageFolderWithPaths

                # Safely get class names using indices
                true_class_name = class_names[true_label_idx] if true_label_idx < len(class_names) else f"Unknown_True_Class_{true_label_idx}"
                predicted_class_name = class_names[predicted_label_idx] if predicted_label_idx < len(class_names) else f"Unknown_Pred_Class_{predicted_label_idx}"

                # Example: Specific misclassifications for 'Glaucoma'
                if true_class_name == 'Glaucoma' and predicted_class_name == 'Myopia':
                    misclassified_glaucoma_as_myopia_paths.append(image_path)
                elif true_class_name == 'Glaucoma' and predicted_class_name == 'Healthy':
                    misclassified_glaucoma_as_healthy_paths.append(image_path)
                # Add more conditions for other critical misclassifications if needed


            if (i + 1) % 10 == 0: # Print progress every 10 batches
                print(f"Processed {i + 1} batches...")

    print('\nEvaluation Complete.')
    
    if len(all_labels) == 0:
        print("No samples were evaluated. Cannot generate report or matrix.")
        return

    print('Classification Report:')
    # target_names maps integer labels to class names for readability
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)

    # Plotting the Confusion Matrix
    if len(class_names) > 0 and len(all_labels) > 0:
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=60, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations for cell values
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]): # Iterate rows (true labels)
            for j in range(cm.shape[1]): # Iterate columns (predicted labels)
                plt.text(j, i, f'{int(cm[i, j])}', horizontalalignment="center", 
                         color="white" if cm[i, j] > thresh else "black", fontsize=8)
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    else:
        print("Cannot plot confusion matrix: No class names or no predictions found to build matrix.")

    # Report misclassified Glaucoma images
    print("\nPaths of Glaucoma images misclassified as Myopia:")
    if misclassified_glaucoma_as_myopia_paths:
        for path in misclassified_glaucoma_as_myopia_paths:
            print(path)
    else:
        print("No Glaucoma images misclassified as Myopia.")

    print("\nPaths of Glaucoma images misclassified as Healthy:")
    if misclassified_glaucoma_as_healthy_paths:
        for path in misclassified_glaucoma_as_healthy_paths:
            print(path)
    else:
        print("No Glaucoma images misclassified as Healthy.")

if __name__ == '__main__':
    # project_root is defined at the top due to sys.path fix
    
    data_folder = os.path.join(project_root, 'data')
    # For final evaluation, you typically use the 'test' set, which the model has NOT seen during training or validation.
    test_data_dir = os.path.join(data_folder, 'test') 

    batch_size = 32
    img_size = (224, 224)
    # Path to the saved model
    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    # Automatically detect and use CUDA (GPU) if available, otherwise fall back to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Attempting to load evaluation data from: {test_data_dir}")

    # Create data loaders for the test set. We only need the test_loader here.
    # include_paths=True is essential for tracking misclassified image file paths.
    # num_workers=0 is safer for Windows to avoid multiprocessing issues, but can be higher on Linux/macOS.
    _, _, test_loader, class_names = create_data_loaders(
        train_dir=None, # Not needed for evaluation
        val_dir=None,   # Not needed for evaluation
        test_dir=test_data_dir,
        batch_size=batch_size,
        img_size=img_size,
        include_paths=True, # Ensure paths are returned for evaluation
        num_workers=0 # Set to 0 if you encounter issues on Windows with data loading
    )

    evaluate_model(model_path, test_loader, class_names, device)