import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import random
from PIL import Image
import torchvision.transforms as transforms

# Import your actual model architecture and data prep functions
from target_model import SimpleCNN
from dataset_prep import prepare_mia_data

# 1. Page Configuration
st.set_page_config(page_title="MIA Analyzer", layout="wide")

nav_col1, nav_col2, nav_col3 = st.columns([4, 1, 1])
with nav_col1:
    st.header("Membership Inference Attack Analyzer")
with nav_col2:
    st.link_button("📄 Read Final Report", "https://drive.google.com/file/d/17VoHHkKLatHDOlzqXfst9439dGDQ2z-q/view?usp=sharing", use_container_width=True)
with nav_col3:
    st.link_button("💻 View on GitHub", "https://github.com/kayshaah/mia", use_container_width=True)
st.divider()

# CIFAR-10 Class Names
classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# 2. Caching Models & Data so the app runs fast
@st.cache_resource
def load_all_assets():
    # Load Attack Model
    attack_model = joblib.load('attack_model.joblib')
    
    # Load Target Model and its trained weights
    target_model = SimpleCNN()
    target_model.load_state_dict(torch.load('target_model.pth', map_location=torch.device('cpu')))
    target_model.eval() # Set to evaluation mode
    
    # Load Datasets
    target_train_set, target_test_set, _, _ = prepare_mia_data()
    
    return attack_model, target_model, target_train_set, target_test_set

attack_model, target_model, target_train_set, target_test_set = load_all_assets()

# Helper function to convert PyTorch tensors back to viewable images
def format_image_for_display(img_tensor):
    img = img_tensor / 2 + 0.5  # Un-normalize from [-1, 1] back to [0, 1]
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0)) # Change from (Color, Height, Width) to (Height, Width, Color)

# 3. UI Layout & Logic
col1, col2 = st.columns([1, 1.5]) # Make the right column slightly wider

with col1:
    st.header("1. Data Selection")
    st.write("Pull an image from the dataset, or upload your own.")
    
    if 'current_image' not in st.session_state:
        st.session_state['current_image'] = None

    # Option A: Pull Training Data
    if st.button("Sample a 'Training Data' Image (Member)", use_container_width=True):
        idx = random.randint(0, len(target_train_set) - 1)
        img, label = target_train_set[idx]
        st.session_state['current_image'] = img
        st.session_state['true_status'] = "Member"
        st.session_state['true_label'] = classes[label]
        
    # Option B: Pull Test Data
    if st.button("Sample a 'New' Image (Non-Member)", use_container_width=True):
        idx = random.randint(0, len(target_test_set) - 1)
        img, label = target_test_set[idx]
        st.session_state['current_image'] = img
        st.session_state['true_status'] = "Non-Member"
        st.session_state['true_label'] = classes[label]

    st.divider()
    
    # Option C: Custom Image Upload
    st.subheader("Or Test Custom Data:")
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Load and convert image to RGB
        user_img = Image.open(uploaded_file).convert('RGB')
        
        # Transform it to 32x32 to match CIFAR-10 exactly
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        st.session_state['current_image'] = transform(user_img)
        st.session_state['true_status'] = "Out-of-Distribution (Non-Member)"
        st.session_state['true_label'] = "Custom Upload"

    # Display the selected image
    if st.session_state['current_image'] is not None:
        st.divider()
        st.subheader("Selected Image:")
        display_img = format_image_for_display(st.session_state['current_image'])
        st.image(np.clip(display_img, 0, 1), width=200, caption=f"True Class: {st.session_state['true_label']}")
        st.info(f"Actual Data Status: **{st.session_state['true_status']}**")

with col2:
    st.header("2. AI Analysis Pipeline")
    
    if st.session_state['current_image'] is not None:
        # Step A: Feed the image to the Target Model
        img_tensor = st.session_state['current_image'].unsqueeze(0) # Add batch dimension
        
        with torch.no_grad():
            raw_output = target_model(img_tensor)
            # Apply Softmax to get probabilities
            probabilities = nn.Softmax(dim=1)(raw_output).numpy()[0] 
            
        # Display the Confidence Chart
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(classes, probabilities, color='darkslateblue')
        ax.set_ylim(0, 1)
        ax.set_ylabel("Confidence")
        ax.set_title("Target Model's Thought Process")
        st.pyplot(fig)
        
        # Step B: Feed the confidence vector to the Attack Model
        vector_to_analyze = probabilities.reshape(1, -1)
        prediction = attack_model.predict(vector_to_analyze)[0]
        attack_confidence = attack_model.predict_proba(vector_to_analyze)[0]
        
        st.divider()
        st.subheader("3. Final Verdict")
        
        # Adding Display Icons for Clarity and Presentation
        if prediction == 1:
            st.error("🚨 **MEMBER DETECTED: PRIVACY LEAK** 🚨")
            st.write(f"The Attack Model is **{attack_confidence[1]*100:.1f}%** sure this image was used in training.")
        else:
            st.success("✅ **NON-MEMBER: SAFE** ✅")
            st.write(f"The Attack Model is **{attack_confidence[0]*100:.1f}%** sure this is brand new data.")
