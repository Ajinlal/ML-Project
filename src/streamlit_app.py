import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# Add project root to sys.path to resolve imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
sys.path.insert(0, project_root)

# Import your model and data loader to get class names
from src.eyecare_model import TransferLearningModel
from src.data_loader import create_data_loaders

# --- Streamlit App Configuration (MUST BE FIRST Streamlit command) ---
st.set_page_config(
    page_title="EyeCare AI Classifier",
    page_icon="🔍",
    layout="centered"
)

# --- Configuration (same as in app.py / predict.py) ---
MODEL_PATH = os.path.join(project_root, 'models', 'best_model.pth')
IMG_SIZE = (224, 224) # Ensure this matches your model's input size
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_ARCHITECTURE = "resnet18" # IMPORTANT: Make sure this matches your trained model!

# --- Load Class Names (needed for mapping model output to labels) ---
# We use create_data_loaders with a dummy path to reliably get class_names.
data_folder = os.path.join(project_root, 'data')
test_data_dir = os.path.join(data_folder, 'test')

# This will correctly load class_names from the folder structure.
# Warnings about 'None' directories are expected here as we only need class_names.
_, _, _, CLASS_NAMES = create_data_loaders(
    train_dir=None, val_dir=None, test_dir=test_data_dir, batch_size=1, img_size=IMG_SIZE, include_paths=False
)

if not CLASS_NAMES:
    st.error("ERROR: Could not determine class names. Please check your data/test directory.")
    st.stop() # Stop the Streamlit app if critical data is missing

# --- Load the Model (only once when the app starts) ---
@st.cache_resource # Cache the model loading for performance
def load_model():
    try:
        num_classes = len(CLASS_NAMES)
        model = TransferLearningModel(num_classes=num_classes, model_name=MODEL_ARCHITECTURE).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        st.success(f"Model '{MODEL_ARCHITECTURE}' loaded successfully on {DEVICE}.")
        return model
    except Exception as e:
        st.error(f"CRITICAL ERROR: Failed to load model from {MODEL_PATH}. "
                 f"Ensure the path is correct and model architecture ({MODEL_ARCHITECTURE}) matches. Error: {e}")
        st.stop() # Stop the app if model cannot be loaded

model = load_model()

# --- Define Preprocessing Transforms (same as for test/validation) ---
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Streamlit App Layout ---
st.title(" EyeCare AI: Retinal Image Classifier")
st.markdown("Upload an eye image to classify it into one of six eye conditions: "
            "**Diabetic Retinopathy, Glaucoma, Healthy, Macular Scar, Myopia, Retinitis Pigmentosa.**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        # Preprocess the image
        image_tensor = preprocess(image)
        image_tensor = image_tensor.unsqueeze(0) # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor.to(DEVICE))
            probabilities = torch.softmax(output, dim=1)

        # Display results
        st.subheader("Prediction Results:")

        # Get top 3 predictions
        top_prob, top_indices = torch.topk(probabilities, k=min(3, len(CLASS_NAMES)))

        for i in range(top_prob.size(1)):
            class_name = CLASS_NAMES[top_indices[0, i].item()]
            confidence = top_prob[0, i].item() * 100
            st.write(f"**{class_name}**: {confidence:.2f}%")

        # Optional: Display all probabilities as a bar chart
        import pandas as pd
        import altair as alt

        df_probs = pd.DataFrame({
            'Class': CLASS_NAMES,
            'Confidence': probabilities.cpu().numpy()[0]
        })
        df_probs = df_probs.sort_values(by='Confidence', ascending=False)

        chart = alt.Chart(df_probs).mark_bar().encode(
            x=alt.X('Confidence', axis=None),
            y=alt.Y('Class', sort='-x'),
            tooltip=['Class', alt.Tooltip('Confidence', format='.2%')]
        ).properties(
            title='All Class Probabilities'
        )
        st.altair_chart(chart, use_container_width=True)


    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("This AI Model can predict EyeDisease with ~93% Accuracy")