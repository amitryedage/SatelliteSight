import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from gradcam_utils import GradCAM

# Config 
MODEL_PATH = r"model_epoch_30.pth" 
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]
CLASS_COLORS = {
    'AnnualCrop': 'green',
    'Forest': 'darkgreen',
    'HerbaceousVegetation': 'lime',
    'Highway': 'orange',
    'Industrial': 'red',
    'Pasture': 'gold',
    'PermanentCrop': 'brown',
    'Residential': 'blue',
    'River': 'skyblue',
    'SeaLake': 'navy'
}
IMG_SIZE = 224

# Load Model 
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image Preprocessing 
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), image

# ---------- UI Setup ----------
st.set_page_config(page_title="Satellite Classifier 🌍", layout="centered")
st.title("🛰️ Satellite Image Classifier with Grad-CAM")
st.markdown("Upload a satellite image (EuroSAT-like) and this app will detect top land types and explain with Grad-CAM.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    with st.spinner("🔍 Analyzing..."):
        input_tensor, original_image = preprocess_image(uploaded_file)
        with torch.no_grad():
            output = model(input_tensor)
            top3_indices = torch.topk(output[0], k=3).indices

        st.markdown("### ✅ Top 3 Predictions:")
        for idx in top3_indices:
            label = CLASS_NAMES[idx]
            color = CLASS_COLORS.get(label, "black")
            st.markdown(f"- <span style='color:{color}; font-weight:bold;'>&#11044; {label}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📸 Grad-CAM Heatmap (Top Class)")

        last_conv_layer = model.layer4[1].conv2  # ResNet18 conv layer
        cam = GradCAM(model, last_conv_layer)
        heatmap = cam.generate(input_tensor)

        # Overlay heatmap on original image
        heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_np = np.array(original_image)
        overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

        st.image(overlay, caption='🔬 Grad-CAM Heatmap (Visual Explanation)', use_container_width=True)
        cam.remove_hooks()
