import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Add project root to sys.path to resolve imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
sys.path.insert(0, project_root)

from src.eyecare_model import TransferLearningModel
from src.data_loader import create_data_loaders # To get class_names

def predict_single_image(model, image_path, img_size, class_names, device, preprocess):
    """
    Predicts the class of a single image using an already loaded model.

    Args:
        model (nn.Module): The loaded PyTorch model.
        image_path (str): Path to the single image file for prediction.
        img_size (tuple): (height, width) expected by the model.
        class_names (list): List of class names corresponding to model outputs.
        device (torch.device): Device to run inference on (e.g., 'cpu' or 'cuda').
        preprocess (torchvision.transforms.Compose): Image preprocessing pipeline.

    Returns:
        tuple: (predicted_class_name, confidence) or (None, None) if prediction fails.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None

    try:
        image = Image.open(image_path).convert('RGB') # Ensure 3 channels
        image_tensor = preprocess(image)
        image_tensor = image_tensor.unsqueeze(0) # Add batch dimension (1, C, H, W)
        
        with torch.no_grad():
            output = model(image_tensor.to(device))
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = class_names[predicted_idx.item()]
        
        print(f"\n--- Prediction Result ---")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence.item():.4f}")
        # print(f"All Probabilities: {probabilities.cpu().numpy()}") # Uncomment to see all probabilities
        print(f"-------------------------\n")

        return predicted_class, confidence.item()

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

if __name__ == '__main__':
    # Configuration
    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    img_size = (224, 224)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_architecture = "resnet18" # IMPORTANT: Make sure this matches your trained model!

    # --- New: Define the folder for images you want to test ---
    inference_image_folder = os.path.join(project_root, 'inference_images')
    
    # Create the folder if it doesn't exist
    if not os.path.exists(inference_image_folder):
        os.makedirs(inference_image_folder)
        print(f"Created inference image folder: {inference_image_folder}")
        print("Please place images you want to test inside this folder and run the script again.")
        sys.exit(0) # Exit if the folder was just created, so user can add images

    # --- Get Class Names ---
    data_folder = os.path.join(project_root, 'data')
    test_data_dir = os.path.join(data_folder, 'test')
    
    _, _, dummy_test_loader, class_names = create_data_loaders(
        train_dir=None, val_dir=None, test_dir=test_data_dir, batch_size=1, img_size=img_size, include_paths=False
    )

    if not class_names:
        print("ERROR: Could not determine class names. Please check your data/test directory.")
        sys.exit(1)

    print(f"Detected class names: {class_names}")

    # --- Load the Model Once ---
    num_classes = len(class_names)
    model = TransferLearningModel(num_classes=num_classes, model_name=model_architecture).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- Define Preprocessing Transforms ---
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Iterate through images in the inference folder ---
    image_files_found = 0
    print(f"\nScanning for images in: {inference_image_folder}")
    for filename in os.listdir(inference_image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            image_path = os.path.join(inference_image_folder, filename)
            print(f"\n--- Processing: {filename} ---")
            predicted_class, confidence = predict_single_image(
                model, image_path, img_size, class_names, device, preprocess
            )
            image_files_found += 1
            if not predicted_class:
                print(f"Skipping {filename} due to an error.")
    
    if image_files_found == 0:
        print(f"No image files found in '{inference_image_folder}'. Please add images to this folder.")
    else:
        print("\n--- All predictions complete ---")